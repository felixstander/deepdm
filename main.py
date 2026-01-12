import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

CONFIG = {
    "seed": 2024,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 20,
    "batch_size": 16,
    "lr": 0.001,
    # --- 新增：词表存储目录 ---
    "vocab_dir": "./vocab_map",
    "sparse_features": ["brand_name", "shop_id", "city_id"],
    "dense_features": ["distance_km", "price_score", "rating"],
    "text_feature_dim": 64,
    # 预留 Buffer 配置
    "vocab_limits": {"brand_name": 500, "shop_id": 1000, "city_id": 50},
    "embed_dim": 64,
}

os.makedirs(CONFIG["vocab_dir"], exist_ok=True)

# 固定随机种子
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])


# ==============================================================================
# 2. 模拟数据生成器 (Fake Data Generator)
# ==============================================================================
def generate_fake_data(num_samples=1000):
    """
    生成模拟数据，为了让模型能学到东西，我们植入一些规则：
    1. 距离近 (distance < 5) 的正样本概率高
    2. 品牌匹配 (如 User是宝马, Shop擅长宝马) 的正样本概率高
    """
    brands = ["宝马", "奔驰", "奥迪", "丰田", "本田", "小米汽车"]  # 小米是后来新增的
    cities = ["上海", "北京", "杭州"]

    data = []
    for _ in range(num_samples):
        # User 特征
        u_brand = random.choice(brands)
        u_city = random.choice(cities)
        u_fault = random.choice(["刹车异响", "无法启动", "漆面划痕", "保养"])

        # Shop 特征
        s_id = random.randint(100, 115)  # 假设只有15家店
        s_specialty = random.choice(brands)  # 擅长修啥
        s_rating = round(random.uniform(3.5, 5.0), 1)

        # Context 特征 (交叉特征)
        distance = round(random.uniform(0.5, 50.0), 1)  # km
        price = round(random.uniform(0.2, 1.0), 2)  # 归一化后的价格

        # --- 构造 Label (模拟真实世界的点击/成交逻辑) ---
        score = 0
        if u_brand == s_specialty:
            score += 0.4  # 品牌匹配加分
        if distance < 10:
            score += 0.3  # 距离近加分
        if s_rating > 4.5:
            score += 0.2  # 评分高加分
        if price < 0.5:
            score += 0.1  # 便宜加分

        # 增加一点随机性
        label = 1 if score + random.uniform(-0.2, 0.2) > 0.5 else 0

        # 构造文本 (用于 BGE)
        # 重点：把品牌拼进去！
        combined_text = f"品牌:{u_brand};故障:{u_fault};网点擅长:{s_specialty}"

        data.append(
            {
                "brand_name": u_brand,
                "shop_id": s_id,
                "city_id": u_city,
                "distance_km": distance,
                "price_score": price,
                "rating": s_rating,
                "text_raw": combined_text,
                "label": label,
            }
        )

    return pd.DataFrame(data)


# ==============================================================================
# 3. 特征处理流水线 (Feature Processing)
# ==============================================================================
class PersistentVocabMapper:
    """
    管理 ID 映射，并将字典持久化保存到 JSON 文件中
    """

    def __init__(self, feature_name, vocab_dir, max_size):
        self.feature_name = feature_name
        self.file_path = os.path.join(vocab_dir, f"{feature_name}.json")
        self.max_size = max_size

        # 初始化默认值
        self.token2id = {"<PAD>": 0, "<UNK>": 1}
        self.next_id = 2

        # 尝试从文件加载
        self._load()

    def _load(self):
        """从 JSON 加载词表"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.token2id = data["token2id"]
                    self.next_id = data["next_id"]
                print(
                    f"[{self.feature_name}] 已加载词表，当前大小: {len(self.token2id)}"
                )
            except Exception as e:
                print(f"[{self.feature_name}] 加载失败，将使用新词表: {e}")

    def _save(self):
        """保存词表到 JSON"""
        data = {"token2id": self.token2id, "next_id": self.next_id}
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # print(f"[{self.feature_name}] 词表已保存 -> {self.file_path}")

    def fit(self, values):
        """训练时使用：扫描数据，更新词表，并保存"""
        has_new = False
        for v in set(values):
            v_str = str(v)  # 强制转字符串，保证 JSON key 一致性
            if v_str not in self.token2id:
                if self.next_id < self.max_size:
                    self.token2id[v_str] = self.next_id
                    self.next_id += 1
                    has_new = True

        if has_new:
            self._save()

    def transform(self, value):
        """推理时使用：支持动态新增"""
        v_str = str(value)

        # 1. 已存在
        if v_str in self.token2id:
            return self.token2id[v_str]

        # 2. 新词，且有空位 -> 动态分配并立即保存
        if self.next_id < self.max_size:
            new_id = self.next_id
            self.token2id[v_str] = new_id
            self.next_id += 1

            # 立即保存，保证下一次请求或其他进程能看到（简单的模拟）
            self._save()
            return new_id

        # 3. 没空位 -> UNK
        else:
            return self.token2id["<UNK>"]


class FeatureProcessor:
    def __init__(self, config):
        self.config = config
        self.mappers = {}
        # 初始化每个稀疏特征的 Mapper
        for feat, limit in config["vocab_limits"].items():
            self.mappers[feat] = PersistentVocabMapper(
                feature_name=feat, vocab_dir=config["vocab_dir"], max_size=limit
            )

    def fit(self, df):
        """扫描训练数据，建立词表"""
        for feat in self.config["sparse_features"]:
            self.mappers[feat].fit(df[feat].values)

    def get_bge_embedding(self, text_list):
        """
        [模拟] 调用 BGE 模型。
        在真实场景下，这里加载 HuggingFace 的模型进行推理。
        这里用随机向量代替，为了让你能跑通代码。
        """
        batch_size = len(text_list)
        # 模拟 BGE 输出 64维向量
        return torch.randn(batch_size, self.config["text_feature_dim"])

    def transform(self, df):
        """将 DataFrame 转化为 Tensor 字典"""
        # 1. 处理 Sparse (Categorical)
        sparse_dict = {}
        for feat in self.config["sparse_features"]:
            # map操作
            ids = [self.mappers[feat].transform(v) for v in df[feat].values]
            sparse_dict[feat] = torch.tensor(ids, dtype=torch.long)

        # 2. 处理 Dense (Numerical) -> 简单归一化
        dense_list = []
        for feat in self.config["dense_features"]:
            vals = df[feat].values.astype(np.float32)
            # 简单的 MaxMin 归一化模拟
            if feat == "distance_km":
                vals = vals / 50.0
            if feat == "rating":
                vals = vals / 5.0
            dense_list.append(vals.reshape(-1, 1))
        dense_tensor = torch.tensor(
            np.concatenate(dense_list, axis=1), dtype=torch.float32
        )

        # 3. 处理 Text (BGE)
        text_tensor = self.get_bge_embedding(df["text_raw"].tolist())

        # 4. Label
        label_tensor = None
        if "label" in df.columns:
            label_tensor = torch.tensor(df["label"].values, dtype=torch.float32).view(
                -1, 1
            )

        return sparse_dict, dense_tensor, text_tensor, label_tensor


# ==============================================================================
# 4. DeepFM 模型定义 (Model Architecture)
# ==============================================================================
class DeepFM(nn.Module):
    def __init__(self, config):
        super(DeepFM, self).__init__()
        self.config = config

        # --- A. Embedding 层 (包含预留 Buffer) ---
        self.embeddings = nn.ModuleDict()
        for feat, limit in config["vocab_limits"].items():
            # 关键：这里申请了 limit 大小的空间 (例如 500)
            self.embeddings[feat] = nn.Embedding(limit, config["embed_dim"])

        # 计算维度
        self.num_sparse = len(config["sparse_features"])
        self.num_dense = len(config["dense_features"])
        self.text_dim = config["text_feature_dim"]

        # Deep 输入总维度 = (Sparse数 * Embed维) + Dense数 + Text维
        deep_input_dim = (
            (self.num_sparse * config["embed_dim"]) + self.num_dense + self.text_dim
        )

        # --- B. Deep Side (DNN) ---
        self.dnn = nn.Sequential(
            nn.Linear(deep_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # --- C. FM Side (简单版：仅做 Embedding 的一阶和二阶) ---
        # FM 一阶 (Sparse 部分)
        self.fm_1st_sparse = nn.ModuleDict()
        for feat, limit in config["vocab_limits"].items():
            self.fm_1st_sparse[feat] = nn.Embedding(limit, 1)

        # FM 一阶 (Dense + Text 部分)
        self.fm_1st_dense = nn.Linear(self.num_dense + self.text_dim, 1)

    def forward(self, sparse_inputs, dense_inputs, text_inputs):
        """
        sparse_inputs: Dict {'brand': [B], ...}
        dense_inputs: [B, N_Dense]
        text_inputs: [B, Text_Dim]
        """

        # 1. 准备 Embeddings
        embed_list = []  # [B, Embed_Dim] 的列表
        for feat in self.config["sparse_features"]:
            tensor = sparse_inputs[feat]
            embed_list.append(self.embeddings[feat](tensor))

        # [B, N_Sparse * Embed_Dim]
        sparse_concat = torch.cat(embed_list, dim=1)

        # 2. 拼接所有特征给 Deep
        # [B, Total_Dim]
        deep_input = torch.cat([sparse_concat, dense_inputs, text_inputs], dim=1)

        # 3. 计算 Deep Logit
        deep_logit = self.dnn(deep_input)

        # 4. 计算 FM Logit
        # 4.1 FM 一阶
        fm_1st_s = sum(
            [
                self.fm_1st_sparse[f](sparse_inputs[f])
                for f in self.config["sparse_features"]
            ]
        )
        fm_1st_d = self.fm_1st_dense(torch.cat([dense_inputs, text_inputs], dim=1))
        fm_1st = fm_1st_s + fm_1st_d

        # 4.2 FM 二阶 (Embedding 交叉)
        # Stack: [B, N_Sparse, Embed_Dim]
        stacked_embeds = torch.stack(embed_list, dim=1)
        sum_square = torch.pow(torch.sum(stacked_embeds, dim=1), 2)
        square_sum = torch.sum(torch.pow(stacked_embeds, 2), dim=1)
        fm_2nd = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)

        # 5. 融合
        total_logit = deep_logit + fm_1st + fm_2nd
        return torch.sigmoid(total_logit)


# ==============================================================================
# 5. 训练 Dataset 封装
# ==============================================================================
class RepairDataset(Dataset):
    def __init__(self, sparse_dict, dense_tensor, text_tensor, label_tensor=None):
        self.sparse_dict = sparse_dict
        self.dense_tensor = dense_tensor
        self.text_tensor = text_tensor
        self.label_tensor = label_tensor
        self.length = len(dense_tensor)
        # 获取 keys 列表以保证顺序
        self.sparse_keys = list(sparse_dict.keys())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 构造单个样本的 sparse 字典
        s_data = {k: self.sparse_dict[k][idx] for k in self.sparse_keys}
        d_data = self.dense_tensor[idx]
        t_data = self.text_tensor[idx]

        if self.label_tensor is not None:
            l_data = self.label_tensor[idx]
            return s_data, d_data, t_data, l_data
        return s_data, d_data, t_data


# ==============================================================================
# 6. 主流程：训练与推理 (Main Pipeline)
# ==============================================================================
def train_pipeline():
    print(">>> 1. 生成模拟数据...")
    df_train = generate_fake_data(1000)
    print(f"    生成 {len(df_train)} 条数据，正样本率: {df_train['label'].mean():.2f}")

    print(">>> 2. 特征工程处理...")
    processor = FeatureProcessor(CONFIG)
    processor.fit(df_train)  # 构建词表

    # 转换数据
    s_dict, d_tensor, t_tensor, l_tensor = processor.transform(df_train)
    dataset = RepairDataset(s_dict, d_tensor, t_tensor, l_tensor)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    print(">>> 3. 初始化 DeepFM 模型...")
    model = DeepFM(CONFIG).to(CONFIG["device"])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    print(">>> 4. 开始训练...")
    model.train()
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        for batch_idx, (s, d, t, label) in enumerate(dataloader):
            # Move to GPU
            s = {k: v.to(CONFIG["device"]) for k, v in s.items()}
            d, t, label = (
                d.to(CONFIG["device"]),
                t.to(CONFIG["device"]),
                label.to(CONFIG["device"]),
            )

            optimizer.zero_grad()
            preds = model(s, d, t)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"    Epoch {epoch+1}/{CONFIG['epochs']}, Avg Loss: {total_loss/len(dataloader):.4f}"
        )

    print(">>> 训练完成！")
    return model, processor


def predict_pipeline(model, processor):
    print("\n========================================")
    print(">>> 5. 模拟线上推理 (包含新品牌)")
    print("========================================")

    model.eval()

    # 模拟一个线上请求，包含一个训练时没见过的品牌 "保时捷"
    # 和一个训练时见过的 "宝马"
    new_requests = [
        {
            "brand_name": "保时捷",  # NEW BRAND!
            "shop_id": 101,
            "city_id": "上海",
            "distance_km": 5.0,
            "price_score": 0.9,  # 贵
            "rating": 4.8,
            "text_raw": "品牌:保时捷;故障:发动机;网点擅长:德系精修",
        },
        {
            "brand_name": "宝马",  # KNOWN BRAND
            "shop_id": 102,
            "city_id": "北京",
            "distance_km": 50.0,  # 远
            "price_score": 0.4,
            "rating": 4.0,
            "text_raw": "品牌:宝马;故障:保养;网点擅长:宝马",
        },
    ]

    df_infer = pd.DataFrame(new_requests)

    # 转换 (注意：Transform 内部有逻辑处理新 ID)
    s_dict, d_tensor, t_tensor, _ = processor.transform(df_infer)

    with torch.no_grad():
        s = {k: v.to(CONFIG["device"]) for k, v in s_dict.items()}
        d, t = d_tensor.to(CONFIG["device"]), t_tensor.to(CONFIG["device"])

        probs = model(s, d, t)

    print("推理结果:")
    for i, req in enumerate(new_requests):
        brand_id = s["brand_name"][i].item()
        print(
            f"客户: {req['brand_name']} (ID映射为: {brand_id}) | 距离: {req['distance_km']}km"
        )
        print(f"-> 推荐得分: {probs[i].item():.4f}")
        if req["brand_name"] == "保时捷":
            print(
                "   (注: 保时捷是新词，分配了新ID，得分主要靠BGE文本向量和数值特征支撑)"
            )
        print("-" * 30)


if __name__ == "__main__":
    trained_model, feature_proc = train_pipeline()
    predict_pipeline(trained_model, feature_proc)
