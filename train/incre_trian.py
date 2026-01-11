import torch
import torch.nn as nn
import torch.optim as optim


def incremental_train(model, new_data_loader, vocab_manager):
    """
    model: 你的 DeepFM 模型 (已加载之前的权重)
    new_data_loader: 最近一天的新增数据 (包含小米汽车的样本)
    """
    model.train()
    
    # 【技巧】为了防止把老品牌的特征“带偏”，可以只针对新 ID 进行大学习率更新
    # 或者简单点，使用较小的全局学习率
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # 学习率调低，微调模式
    
    for batch in new_data_loader:
        # batch_brands: [Batch, 1] -> 包含 ID 105
        dense, sparse, bge_vecs, labels = batch
        
        # 前向传播
        preds = model(dense, sparse, bge_vecs)
        loss = nn.BCELoss()(preds, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        # 此时，ID=105 对应的 Embedding 向量会从随机值迅速收敛到有意义的值
        # 因为 BGE 向量作为辅助，告诉模型 "小米" 和 "电车" 相似，
        # 梯度下降会引导 Embedding 向量也往那个方向走。
        optimizer.step()
        
    # 保存微调后的模型
    torch.save(model.state_dict(), "model_v2.pth")
    print("增量训练完成，新品牌 Embedding 已更新。")
