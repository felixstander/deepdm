import copy
import datetime
import json
import os
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config.basic_config import CONFIG
from feature.feat_process import FeatureProcessor, RepairDataset
from model.model import DeepFM


def train(config: Dict, train_data: pd.DataFrame):
    print(">>> 1. 划分训练集和验证集...")
    train_df, val_df = train_test_split(
        train_data,
        test_size=0.2,
        random_state=config["seed"],
        stratify=train_data["brand_name"],
    )
    print(
        f"    总共 {len(train_data)} 条数据, "
        f"训练集: {len(train_df)} 条 (正样本率: {train_df['label'].mean():.2f}), "
        f"验证集: {len(val_df)} 条 (正样本率: {val_df['label'].mean():.2f})"
    )

    print(">>> 2. 特征工程处理...")
    processor = FeatureProcessor(config)
    processor.fit(train_df)  # 使用训练集构建词表

    # 转换训练集
    s_dict_train, d_tensor_train, t_tensor_train, l_tensor_train = processor.transform(
        train_df
    )
    train_dataset = RepairDataset(
        s_dict_train, d_tensor_train, t_tensor_train, l_tensor_train
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    # 转换验证集
    s_dict_val, d_tensor_val, t_tensor_val, l_tensor_val = processor.transform(val_df)
    val_dataset = RepairDataset(s_dict_val, d_tensor_val, t_tensor_val, l_tensor_val)
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    print(">>> 3. 初始化 DeepFM 模型...")
    model = DeepFM(config).to(config["device"])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    print(">>> 4. 开始训练...")
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(config["epochs"]):
        # --- 训练模式 ---
        model.train()
        total_train_loss = 0
        for batch_idx, (s, d, t, label) in enumerate(train_dataloader):
            s = {k: v.to(config["device"]) for k, v in s.items()}
            d, t, label = (
                d.to(config["device"]),
                t.to(config["device"]),
                label.to(config["device"]),
            )

            optimizer.zero_grad()
            preds = model(s, d, t)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # --- 评估模式 ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for s, d, t, label in val_dataloader:
                s = {k: v.to(config["device"]) for k, v in s.items()}
                d, t, label = (
                    d.to(config["device"]),
                    t.to(config["device"]),
                    label.to(config["device"]),
                )
                preds = model(s, d, t)
                loss = criterion(preds, label)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        print(
            f"    Epoch {epoch+1}/{config['epochs']}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"    -> New best model saved with Val Loss: {best_val_loss:.4f}")

    print(">>> 训练完成！")

    # --- 保存最佳模型权重和配置 ---
    if best_model_state:
        print(">>> 5. 保存最佳模型权重和线上配置...")
        history_model_dir = config["history_model_dir"]
        serving_conf_path = config["serving_conf_path"]
        os.makedirs(history_model_dir, exist_ok=True)
        os.makedirs(os.path.dirname(serving_conf_path), exist_ok=True)


        current_date_str = datetime.datetime.now()
        today_date_str = datetime.datetime.strftime(current_date_str, "%Y-%m-%d")
        save_name = f"model_{today_date_str}.pth"
        save_path = os.path.join(history_model_dir, save_name)
        torch.save(best_model_state, save_path)

        config_data = {
            "active_model_path": save_path,
            "updated_at": str(datetime.datetime.now()),
            "metrics": {"auc": "", "loss": best_val_loss},
        }
        with open(serving_conf_path, "w") as f:
            json.dump(config_data, f, indent=2)

        print(">>> 模型权重和线上配置完成！")
    else:
        print(">>> 训练未产出有效模型，请检查数据或参数。")


if __name__ == "__main__":
    data_path = './data/generate_fake_data.csv'
    train_data = pd.read_csv(data_path)
    train(config=CONFIG, train_data=train_data)
