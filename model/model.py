import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, feature_columns, hidden_units=(128, 64), dropout=0.2):
        """
        feature_columns: 配置字典，定义每种特征的类型和维度
        """
        super(DeepFM, self).__init__()
        
        # ==========================================
        # 1. 特征定义部分
        # ==========================================
        self.dense_feature_cols = feature_columns['dense'] # 数值特征列表
        self.sparse_feature_cols = feature_columns['sparse'] # 类别特征配置
        
        # 稀疏特征 (Category) 的 Embedding 层
        # 作用：既用于 FM 的隐向量，也用于 Deep 部分的输入
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['vocab_size'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })
        
        # 计算 Deep 部分的总输入维度
        # Dense特征数 + Sum(Embedding维度) + BGE向量维度
        self.dense_input_dim = len(self.dense_feature_cols) 
        self.sparse_input_dim = sum([feat['embed_dim'] for feat in self.sparse_feature_cols])
        # 假设 BGE 向量是 64维 (User) + 64维 (Shop) = 128
        self.bge_dim = 128 
        
        deep_input_dim = self.dense_input_dim + self.sparse_input_dim + self.bge_dim

        # ==========================================
        # 2. FM 部分 (一阶 + 二阶)
        # ==========================================
        # FM 一阶 (Linear Part): 类似于逻辑回归，对每个特征赋予一个权重 w
        # 针对稀疏特征，用 Embedding(1) 来模拟权重
        self.fm_1st_sparse = nn.ModuleDict({
            'w_' + str(i): nn.Embedding(feat['vocab_size'], 1)
            for i, feat in enumerate(self.sparse_feature_cols)
        })
        # 针对数值特征，直接用 Linear
        self.fm_1st_dense = nn.Linear(self.dense_input_dim + self.bge_dim, 1)

        # ==========================================
        # 3. Deep 部分 (DNN)
        # ==========================================
        layers = []
        input_dim = deep_input_dim
        for unit in hidden_units:
            layers.append(nn.Linear(input_dim, unit))
            layers.append(nn.BatchNorm1d(unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = unit
        layers.append(nn.Linear(input_dim, 1))
        self.deep_layers = nn.Sequential(*layers)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, dense_inputs, sparse_inputs, bge_inputs):
        """
        dense_inputs: [Batch, N_Dense] (包含: 导航距离, 价格, 评分, 品牌匹配Flag)
        sparse_inputs: [Batch, N_Sparse] (包含: 品牌ID, 城市ID, 资质等级ID)
        bge_inputs: [Batch, 128] (包含: User文本向量 + Shop文本向量)
        """
        
        # --- A. 准备 Embedding 向量 ---
        sparse_embeds = [] # [Batch, Embed_Dim] 列表
        for i, feat in enumerate(self.sparse_feature_cols):
            # 取出对应的列
            val = sparse_inputs[:, i].long()
            embed = self.embed_layers['embed_' + str(i)](val)
            sparse_embeds.append(embed)
            
        # 将所有 Embedding 拼接: [Batch, Total_Sparse_Dim]
        sparse_embeds_concat = torch.cat(sparse_embeds, dim=1)
        
        # 将 数值 + BGE 拼接
        dense_all = torch.cat([dense_inputs, bge_inputs], dim=1)

        # ==========================================
        # Part 1: FM 一阶项 (Linear)
        # ==========================================
        fm_1st_sparse_res = [self.fm_1st_sparse['w_' + str(i)](sparse_inputs[:, i].long()) 
                             for i in range(len(self.sparse_feature_cols))]
        fm_1st_sparse_sum = torch.sum(torch.cat(fm_1st_sparse_res, dim=1), dim=1, keepdim=True)
        
        fm_1st_dense_sum = self.fm_1st_dense(dense_all)
        
        fm_1st_part = fm_1st_sparse_sum + fm_1st_dense_sum
        
        # ==========================================
        # Part 2: FM 二阶项 (Interaction)
        # ==========================================
        # 传统的 FM 二阶主要针对 Embedding 之间的交叉
        # 核心公式: 0.5 * (sum(embed)^2 - sum(embed^2))
        
        # 堆叠成 [Batch, Num_Fields, Embed_Dim] 
        # 注意：这里要求不同特征的 Embedding 维度最好一致，如果不一致需要 Padding 或只对一致的做交叉
        # 为了简化，假设主要对 Brand, City 等核心ID做交叉
        # 这里仅作示意，实际工程中通常会把 Embedding 维度统一设为 16
        stacked_embeds = torch.stack(sparse_embeds, dim=1) 
        
        sum_square = torch.pow(torch.sum(stacked_embeds, dim=1), 2)
        square_sum = torch.sum(torch.pow(stacked_embeds, 2), dim=1)
        
        fm_2nd_part = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        
        # ==========================================
        # Part 3: Deep 部分 (DNN)
        # ==========================================
        # 输入：[Sparse_Embeddings, Dense_Features, BGE_Vectors]
        deep_input = torch.cat([sparse_embeds_concat, dense_all], dim=1)
        deep_out = self.deep_layers(deep_input)
        
        # ==========================================
        # Part 4: 最终融合
        # ==========================================
        total_logit = fm_1st_part + fm_2nd_part + deep_out
        return self.sigmoid(total_logit)
