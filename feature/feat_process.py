import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


# ==============================================================================
# 特征处理流水线 (Feature Processing)
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
# 封装 Dataset
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
