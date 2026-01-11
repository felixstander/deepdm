import torch

from redis_cache.online_vocab import OnlineVocabHandler

# --- 全局初始化 (只执行一次) ---
# 假设模型预留了 500 个位置
vocab_handler = OnlineVocabHandler(model_vocab_limit=500) 

# 加载模型
model = DeepFM(...) 
model.load_state_dict(torch.load("model.pth"))
model.eval()

def predict(request_json):
    """
    request_json: {"car_brand": "小米汽车", "text": "...", "price": 0.8}
    """
    brand_name = request_json.get("car_brand")
    
    # =================================================
    # 核心调用点：这里会自动处理新旧品牌
    # =================================================
    brand_id = vocab_handler.get_brand_id(brand_name)
    # =================================================
    
    # 构造 Tensor
    # sparse_inputs: [Batch=1, 1]
    sparse_tensor = torch.tensor([[brand_id]], dtype=torch.long)
    
    # 处理其他特征 (BGE, Dense...)
    dense_tensor = ...
    bge_tensor = ... 
    
    # 推理
    with torch.no_grad():
        score = model(dense_tensor, sparse_tensor, bge_tensor)
        
    return {"score": score.item(), "matched_brand_id": brand_id}
