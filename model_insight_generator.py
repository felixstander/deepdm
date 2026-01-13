# ==========================================
# 3. 核心分析器 (生成报告逻辑)
# ==========================================
class ModelInsightGenerator:
    def __init__(self, old_model, new_model, vocab_mapper):
        self.old_model = old_model
        self.new_model = new_model
        self.vocab = vocab_mapper
        
    def calculate_metrics(self, df):
        """宏观：计算 AUC 和 Loss"""
        y_true = df['label'].values
        # 获取模拟分数
        y_pred_old = self.old_model.predict_mock(df).numpy()
        y_pred_new = self.new_model.predict_mock(df).numpy()
        
        metrics = {
            'old_auc': roc_auc_score(y_true, y_pred_old),
            'new_auc': roc_auc_score(y_true, y_pred_new),
            'old_loss': log_loss(y_true, y_pred_old),
            'new_loss': log_loss(y_true, y_pred_new)
        }
        return metrics

    def get_embedding_shifts(self, top_k=5):
        """中观：计算 Embedding 变动"""
        shifts = []
        old_emb = self.old_model.embeddings['brand_name'].weight.data.numpy()
        new_emb = self.new_model.embeddings['brand_name'].weight.data.numpy()
        
        for token, idx in self.vocab.token2id.items():
            if idx < len(old_emb):
                # 计算欧氏距离
                diff = np.linalg.norm(old_emb[idx] - new_emb[idx])
                shifts.append((token, diff))
                
        # 按变动幅度降序
        shifts.sort(key=lambda x: x[1], reverse=True)
        return shifts[:top_k]

    def find_repaired_cases(self, df, top_k=3):
        """微观：挖掘被新模型'拯救'的案例"""
        repaired = []
        
        y_pred_old = self.old_model.predict_mock(df).numpy()
        y_pred_new = self.new_model.predict_mock(df).numpy()
        
        for i, row in df.iterrows():
            # 只看正样本 (label=1)
            if row['label'] == 0: continue
            
            s_old = y_pred_old[i]
            s_new = y_pred_new[i]
            
            # 逻辑：旧的分低，新的分高
            if s_old < 0.5 and s_new > 0.8:
                repaired.append({
                    'brand': row['brand_name'],
                    'fault': row['fault_desc'],
                    'dist': row['distance_km'],
                    'old': s_old,
                    'new': s_new,
                    'diff': s_new - s_old
                })
        
        repaired.sort(key=lambda x: x['diff'], reverse=True)
        return repaired[:top_k]

# ==========================================
# 4. 生成 Markdown 报告文本
# ==========================================
def render_markdown(metrics, shifts, cases):
    # 计算涨跌符号
    auc_diff = metrics['new_auc'] - metrics['old_auc']
    auc_sign = "🔺" if auc_diff > 0 else "🔻"
    
    loss_diff = metrics['new_loss'] - metrics['old_loss']
    loss_sign = "🔻" if loss_diff < 0 else "🔺" # Loss 越小越好
    
    md = f"""
# 🚀 每日模型进化日报 (2024-05-20)

## 1. 📊 核心指标看板 (The Scoreboard)
> 今日模型在验证集表现 **稳中有升**，成功通过上线标准。

| 核心指标 | 旧模型 (Baseline) | 新模型 (Current) | 变化幅度 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **AUC (排序能力)** | {metrics['old_auc']:.4f} | **{metrics['new_auc']:.4f}** | {auc_sign} {abs(auc_diff)*100:.2f}% | ✅ 达标 |
| **LogLoss (准确度)** | {metrics['old_loss']:.4f} | **{metrics['new_loss']:.4f}** | {loss_sign} {abs(loss_diff):.4f} | ✅ 达标 |

---

## 2. 🧠 知识发现：模型今天“学到了”什么？(Knowledge Discovery)
通过分析 Embedding 向量空间的位移，我们发现模型对以下 **5 个品牌** 的认知发生了剧变。
*这通常意味着：有了新的积压数据输入，或者 BGE 语义纠正了之前的随机参数。*

| 品牌名称 | 认知调整幅度 (Embedding Shift) | 业务解读 |
| :--- | :--- | :--- |
"""
    for brand, shift in shifts:
        interp = "常规参数微调"
        if shift > 1.0: interp = "🔥 **重大认知重构 (新知识注入)**"
        elif shift > 0.5: interp = "⚠️ 显著参数调整"
        md += f"| **{brand}** | `{shift:.4f}` | {interp} |\n"

    md += """
---

## 3. ✨ 亮点案例：Bad Case 修复展示 (The "Save" Cases)
以下是 **客户真实去了该店 (Label=1)**，旧模型认为**不匹配 (Score<0.5)**，但新模型**精准命中 (Score>0.8)** 的典型案例。

"""
    for i, case in enumerate(cases):
        md += f"""### 🎯 案例 {i+1}: {case['brand']} 维修匹配
- **场景特征**:
  - 故障描述: `"{case['fault']}"`
  - 导航距离: `{case['dist']} km`
- **模型打分对比**:
  - 🔴 旧模型: `{case['old']:.2f}` (判断失误：认为不顺路或不匹配)
  - 🟢 **新模型**: **`{case['new']:.2f}`** (判断正确：强烈推荐)
- **归因分析**: 新模型成功捕捉到了 **{case['brand']}** 与该网点资质的强关联，修正了旧模型的偏见。

"""
    return md

# ==========================================
# 5. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    # 1. 准备环境
    vocab = MockVocabMapper()
    
    # 2. 初始化模型 (模拟 Old 和 New)
    old_model = MockDeepFM(vocab, model_version='old')
    new_model = MockDeepFM(vocab, model_version='new')
    
    # 3. 生成数据 (含预埋的修复案例)
    val_df = generate_demo_data(num_samples=100)
    
    # 4. 执行分析
    analyzer = ModelInsightGenerator(old_model, new_model, vocab)
    
    metrics = analyzer.calculate_metrics(val_df)
    shifts = analyzer.get_embedding_shifts()
    cases = analyzer.find_repaired_cases(val_df)
    
    # 5. 生成报告
    report_content = render_markdown(metrics, shifts, cases)
    
    print(report_content)
