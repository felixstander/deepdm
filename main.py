import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config.basic_config import CONFIG

os.makedirs(CONFIG["vocab_dir"], exist_ok=True)

# 固定随机种子
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])




# ==============================================================================
# 主流程：训练与推理 (Main Pipeline)
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
