import datetime
import json
import os
from datetime import timedelta

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

# 假设这些类在 deepfm_persistent.py 中，请确保该文件在同一目录下
# from deepfm_persistent import DeepFM, FeatureProcessor, RepairDataset

class DailyIncrementalPipeline:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        # --- 1. 目录结构初始化 ---
        self.history_model_dir = './model_history'          # 模型权重历史
        self.serving_conf_path = './serving_config/active_model.json' # 线上配置
        
        # 数据存档目录 (按日期归档用)
        self.history_val_dir = './data/history_val'         # 每日验证集归档
        self.history_train_dir = './data/history_train'     # 每日训练集归档
        
        # 积压数据目录 (用于重训)
        self.buffer_dir = './data/buffer'                   # 存放效果不好时的积压数据
        self.buffer_file = os.path.join(self.buffer_dir, 'accumulated_train_buffer.csv')
        
        # 创建所有目录
        for path in [self.history_model_dir, os.path.dirname(self.serving_conf_path), 
                     self.history_val_dir, self.history_train_dir, self.buffer_dir]:
            os.makedirs(path, exist_ok=True)

    def get_current_best_model_path(self):
        """获取当前线上正在使用的模型路径"""
        if os.path.exists(self.serving_conf_path):
            try:
                with open(self.serving_conf_path, 'r') as f:
                    conf = json.load(f)
                    path = conf.get('active_model_path')
                    if path and os.path.exists(path):
                        return path
            except Exception as e:
                print(f"    [Error] 读取线上配置失败: {e}")
        return None

    def load_and_sample_history_val(self, current_date_str, lookback_days=7, sample_frac=0.2):
        """回放历史验证集，防止遗忘 (保持不变)"""
        history_dfs = []
        curr_date = datetime.datetime.strptime(current_date_str, "%Y-%m-%d")
        
        for i in range(1, lookback_days + 1):
            prev_date = curr_date - timedelta(days=i)
            prev_date_str = prev_date.strftime("%Y-%m-%d")
            file_path = os.path.join(self.history_val_dir, f"val_{prev_date_str}.csv")
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    history_dfs.append(df)
                except Exception:
                    pass
        
        if not history_dfs:
            return None
            
        full_history_df = pd.concat(history_dfs, ignore_index=True)
        # 分层采样，保证覆盖所有品牌
        try:
            sampled_df = full_history_df.groupby('brand_name', group_keys=False).apply(
                lambda x: x.sample(frac=sample_frac, random_state=2024) if len(x) > 5 else x.sample(n=min(len(x), 1))
            )
            print(f"    [History Val] 回溯历史验证集: {len(sampled_df)} 条")
            return sampled_df
        except Exception as e:
            print(f"    [Warning] 历史验证集采样失败，使用全量: {e}")
            return full_history_df.sample(frac=sample_frac)

    def evaluate(self, model, dataloader):
        """评估模型 AUC 和 Loss"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for s, d, t, l in dataloader:
                s = {k: v.to(self.device) for k, v in s.items()}
                d, t = d.to(self.device), t.to(self.device)
                
                preds = model(s, d, t)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(l.cpu().numpy())
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
            loss = log_loss(all_labels, all_preds)
        except ValueError:
            # 极少情况：验证集只有正样本或只有负样本
            auc = 0.5
            loss = 99.9
            
        return auc, loss

    def run_daily_job(self, today_date_str, daily_df):
        print(f"\n========== 开始执行每日增量训练: {today_date_str} ==========")
        
        # =========================================================
        # Step 1: 数据切分与归档 (Split & Archive)
        # =========================================================
        print(">>> [Step 1] 数据处理与归档...")
        
        #TODO:当日的数量出了按时间进行切分外，其实还得按照其他维度，比如说车品牌、距离等，这个明天再来考虑
        if 'timestamp' in daily_df.columns:
            daily_df = daily_df.sort_values('timestamp')
            
        split_idx = int(len(daily_df) * 0.8)
        today_raw_train = daily_df.iloc[:split_idx].copy() # 今日新增训练
        today_raw_val = daily_df.iloc[split_idx:].copy()   # 今日新增验证
        
        # 1.1 强制保存今日验证集 (Requirement: 验证集每一天都要保存)
        val_save_path = os.path.join(self.history_val_dir, f"val_{today_date_str}.csv")
        today_raw_val.to_csv(val_save_path, index=False)
        print(f"    [Archive] 今日验证集已保存: {val_save_path}")
        
        # 1.2 强制保存今日训练集 (备份用)
        train_save_path = os.path.join(self.history_train_dir, f"train_{today_date_str}.csv")
        today_raw_train.to_csv(train_save_path, index=False)
        print(f"    [Archive] 今日训练集已备份: {train_save_path}")

        # =========================================================
        # Step 2: 构造最终训练集 (检查积压 Buffer)
        # =========================================================
        print(">>> [Step 2] 检查积压数据...")
        
        final_train_df = today_raw_train
        has_buffer_data = False
        
        if os.path.exists(self.buffer_file):
            try:
                buffer_df = pd.read_csv(self.buffer_file)
                if not buffer_df.empty:
                    print(f"    发现积压数据 {len(buffer_df)} 条，合并训练！")
                    final_train_df = pd.concat([buffer_df, today_raw_train], ignore_index=True)
                    has_buffer_data = True
            except Exception as e:
                print(f"    [Warning] 读取积压数据失败: {e}")
        
        print(f"    最终训练集规模: {len(final_train_df)} 条")

        # =========================================================
        # Step 3: 构造最终验证集 (今日 + 历史回放)
        # =========================================================
        history_val_sample = self.load_and_sample_history_val(today_date_str)
        if history_val_sample is not None:
            final_val_df = pd.concat([today_raw_val, history_val_sample], ignore_index=True)
        else:
            final_val_df = today_raw_val
            
        print(f"    最终验证集规模: {len(final_val_df)} 条")

        # =========================================================
        # Step 4: 特征工程 (Transform)
        # =========================================================
        processor = FeatureProcessor(self.config)
        # 注意：要在最终的合并训练集上 fit，确保积压数据里的生僻词也被包含
        processor.fit(final_train_df) 
        
        train_s, train_d, train_t, train_l = processor.transform(final_train_df)
        val_s, val_d, val_t, val_l = processor.transform(final_val_df)
        
        train_dataset = RepairDataset(train_s, train_d, train_t, train_l)
        val_dataset = RepairDataset(val_s, val_d, val_t, val_l)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)

        # =========================================================
        # Step 5: 加载旧模型 & 评估基准 (Baseline Eval)
        # =========================================================
        print(">>> [Step 5] 加载模型 & 评估基准...")
        model = DeepFM(self.config).to(self.device)
        best_model_path = self.get_current_best_model_path()
        
        old_auc = 0.5
        if best_model_path:
            print(f"    加载权重: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
            # 评估旧模型在当前混合验证集上的表现
            old_auc, old_loss = self.evaluate(model, val_loader)
            print(f"    [Old Model] AUC: {old_auc:.4f} | Loss: {old_loss:.4f}")
        else:
            print("    [Cold Start] 无历史模型，基准 AUC 设为 0.5")

        # =========================================================
        # Step 6: 增量训练 (Training Loop)
        # =========================================================
        print(">>> [Step 6] 执行训练...")
        model.train()
        # 增量训练使用较小学习率
        optimizer = optim.Adam(model.parameters(), lr=1e-4) 
        criterion = nn.BCELoss()
        
        # 仅训练 1 个 Epoch (增量训练不宜过多)
        for s, d, t, l in train_loader:
            s = {k: v.to(self.device) for k, v in s.items()}
            d, t, l = d.to(self.device), t.to(self.device), l.to(self.device)
            
            optimizer.zero_grad()
            preds = model(s, d, t)
            loss = criterion(preds, l)
            loss.backward()
            optimizer.step()

        # =========================================================
        # Step 7: 评估新模型 (New Eval)
        # =========================================================
        new_auc, new_loss = self.evaluate(model, val_loader)
        print(f"    [New Model] AUC: {new_auc:.4f} | Loss: {new_loss:.4f}")

        # =========================================================
        # Step 8: 决策与后续处理 (Decision Logic)
        # =========================================================
        print(">>> [Step 8] 决策...")
        
        # 判断逻辑：AUC 提升，或者冷启动
        if not best_model_path or new_auc >= old_auc:
            # --- Case A: 效果好，保存模型，清空积压 ---
            print(f"    [SUCCESS] 效果提升 (New {new_auc:.4f} >= Old {old_auc:.4f})")
            
            # 1. 保存模型权重
            save_name = f"model_{today_date_str}.pth"
            save_path = os.path.join(self.history_model_dir, save_name)
            torch.save(model.state_dict(), save_path)
            
            # 2. 更新线上配置
            config_data = {
                "active_model_path": save_path,
                "updated_at": str(datetime.datetime.now()),
                "metrics": {"auc": new_auc, "loss": new_loss}
            }
            with open(self.serving_conf_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            # 3. 清空积压数据 (因为已经学会了)
            if os.path.exists(self.buffer_file):
                os.remove(self.buffer_file)
                print("    [Buffer] 积压数据已成功学习，Buffer 已清空。")
                
            print(f"    模型已上线: {save_path}")
            
        else:
            # --- Case B: 效果不好，丢弃权重，保存积压 ---
            print(f"    [FAIL] 效果未提升 (New {new_auc:.4f} < Old {old_auc:.4f})")
            print("    [Rollback] 放弃本次模型更新。")
            
            # (Requirement: 如果效果没有提升，就把训练集保存下来)
            # 我们将本次使用的全量训练集 (Buffer + Today) 全部写回 Buffer
            # 这样明天训练时，会再次包含这些数据
            final_train_df.to_csv(self.buffer_file, index=False)
            print(f"    [Buffer] 训练数据已保存至积压区: {self.buffer_file}")
            print(f"             当前积压池大小: {len(final_train_df)} 条，等待明日合并训练。")

if __name__ == "__main__":
    # 假设你已经有了 CONFIG 和 generate_fake_data 函数 (来自上一段代码)
    # 此处省略 CONFIG 定义，直接使用上一段的 CONFIG
    
    # 1. 模拟第一天 (冷启动)
    pipeline = DailyIncrementalPipeline(CONFIG)
    
    # 模拟生成 "2024-01-01" 的数据
    df_day1 = generate_fake_data(num_samples=1000) 
    pipeline.run_daily_job("2024-01-01", df_day1)
    
    # 2. 模拟第二天 (增量更新)
    # 模拟生成 "2024-01-02" 的数据，包含一些新模式
    df_day2 = generate_fake_data(num_samples=1000)
    pipeline.run_daily_job("2024-01-02", df_day2)
