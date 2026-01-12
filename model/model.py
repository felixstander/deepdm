import torch
import torch.nn as nn

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
