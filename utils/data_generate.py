
import random

import pandas as pd


# ==============================================================================
# 模拟数据生成器 (Fake Data Generator)
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

if __name__ == "__main__":
    data = generate_fake_data(50000)
    save_path = './data/generate_fake_data.csv'
    data.to_csv(save_path,index=False)
