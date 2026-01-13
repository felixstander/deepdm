CONFIG = {
    "seed": 2024,
    "device": "cpu",#cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 20,
    "batch_size": 16,
    "lr": 0.001,
    # --- 新增：词表存储目录 ---
    "vocab_dir": "./vocab_map",
    "history_model_dir":"./model_history",
    "serving_conf_path":"./config/serving_config/active_model.json",# 线上配置
    # 数据存档目录 (按日期归档用)
    "history_val_dir":"./data/history_val",# 每日验证集归档
    "history_train_dir":"./data/history_train",# 每日训练集归档
    "buffer_dir":"./data/buffer",# 存放效果不好时的积压数据
    "sparse_features": ["brand_name", "shop_id", "city_id"],
    "dense_features": ["distance_km", "price_score", "rating"],
    "text_feature_dim": 768,
    # 预留 Buffer 配置
    "vocab_limits": {"brand_name": 500, "shop_id": 1000, "city_id": 50},
    "embed_dim": 128,
}
