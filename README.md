# DeepDM - 深度学习推荐模型演示

本项目展示了一个完整的、端到端的深度学习推荐模型（DeepFM）流水线，从数据模拟到线上推理。它特别强调了在真实世界工程中所面临的挑战，如增量训练、模型稳定性以及在不中断服务的情况下动态处理新特征（例如新品牌）的能力。

## 项目结构

```
/
├─── main.py               # 主脚本，整合所有模块以进行全流程运行
├─── inference.py          # 线上模型预测的入口
├─── pyproject.toml        # 项目依赖与配置
├─── README.md             # 本文件
├─── feature/              # 特征工程脚本
│    ├─── feat_process.py   # 特征词表管理的核心逻辑
│    └─── feat_transform.py # 应用特征转换的脚本
├─── model/                # 模型架构定义
│    └─── model.py          # DeepFM 模型实现
├─── redis_cache/          # 与 Redis 缓存交互的代码
│    └─── online_vocab.py   # 在线管理动态特征词表
├─── train/                # 训练脚本
│    ├─── incre_train.py    # 用于每日更新的增量训练脚本
│    └─── init_train.py     # 用于初次模型部署的全量训练脚本
├─── utils/                # 工具类脚本
│    └─── data_generate.py  # 生成模拟数据的脚本
└─── vocab_map/            # 存放特征词表的目录
     ├─── brand_name.json   # 品牌名称词表
     ├─── city_id.json      # 城市ID词表
     └─── shop_id.json      # 店铺ID词表
```

### 关键组件

- **`main.py`**: 一个集成的脚本，演示了整个离线工作流：数据生成 -> 特征工程 -> 模型训练 -> 模拟预测。这是理解整体流程的最佳起点。
- **`model/model.py`**: 定义了 `DeepFM` 网络架构，它结合了深度神经网络（用于高阶特征交叉）和因子分解机（用于低阶特征交叉）。
- **`train/`**: 包含核心的训练逻辑。
  - `init_train.py`: 全量训练脚本，用于初次模型部署或周期性重训。
  - `incre_train.py`: 增量训练脚本，为每日定时任务设计。
- **`inference.py`**: 线上推理的入口，接收请求、处理特征并返回模型预测结果。
- **`redis_cache/`**: 存放与 Redis 交互的代码，主要用于在线上推理时动态查询和更新特征词表。
- **`utils/`**: 包含工具类脚本，如数据生成脚本。
- **`vocab_map/`**: 存放离线训练时生成的特征词表（ID映射表），以 JSON 格式存储，便于版本控制和追溯。

## 核心流程

项目遵循经典的“离线训练 -> 线上推理”模式，并加入了增量学习和特征动态扩容的关键能力。

### 1. 数据生成 (模拟)

通过脚本生成模拟的业务数据，包括用户特征、网点特征、上下文以及最终的标签（是否推荐）。

**代码示例 (`main.py`):**
```python
# 在 main.py 中
def generate_fake_data(num_samples=1000):
    brands = ["宝马", "奔驰", "奥迪", "丰田", "本田", "小米汽车"]
    cities = ["上海", "北京", "杭州"]
    data = []
    for _ in range(num_samples):
        # ... 模拟生成各种特征 ...
        data.append({
            "brand_name": u_brand,
            "shop_id": s_id,
            "city_id": u_city,
            "distance_km": distance,
            "price_score": price,
            "rating": s_rating,
            "text_raw": combined_text,
            "label": label,
        })
    return pd.DataFrame(data)

# 执行
df_train = generate_fake_data(1000)
print(df_train.head())
```

### 2. 特征工程 & 词表管理

在训练前，需要对原始数据进行处理，特别是将类别特征（如品牌、城市）转换为模型可以接受的 ID。

- **`PersistentVocabMapper`**: 这个类负责管理每个类别特征的 ID 映射。
- **持久化**: 它会将生成的 `token -> id` 映射表保存为 JSON 文件（在 `vocab_map/` 目录下），确保每次训练都能加载和更新同一份词表。
- **动态扩容**: 词表被设计为有容量上限（`max_size`）。在训练或推理时遇到新词，只要词表未满，就会自动为其分配新 ID 并更新 JSON 文件。

**代码示例 (`main.py`):**
```python
# 在 main.py 中
class PersistentVocabMapper:
    # ... (实现见源码) ...

class FeatureProcessor:
    def __init__(self, config):
        self.config = config
        self.mappers = {
            feat: PersistentVocabMapper(
                feature_name=feat, 
                vocab_dir=config["vocab_dir"], 
                max_size=limit
            )
            for feat, limit in config["vocab_limits"].items()
        }

    def fit(self, df):
        """扫描训练数据，建立/更新词表"""
        for feat in self.config["sparse_features"]:
            self.mappers[feat].fit(df[feat].values)

    def transform(self, df):
        """将 DataFrame 转换为模型输入的 Tensors"""
        # ... (实现见源码) ...

# --- 执行流程 ---
# 1. 初始化 Processor
processor = FeatureProcessor(CONFIG)

# 2. 在训练数据上拟合，生成/更新 vocab_map/*.json
processor.fit(df_train)

# 3. 将数据转换为 Tensor
sparse_dict, dense_tensor, text_tensor, label_tensor = processor.transform(df_train)
```

### 3. 模型训练

模型训练分为两种模式：
- **首次/全量训练 (`init_train.py`)**: 使用全量历史数据进行训练，通常在项目启动或模型效果严重衰退时执行。
- **增量训练 (`incre_train.py`)**: 每日定时任务，使用当天的新增数据对前一天的模型进行微调，以快速适应新数据分布。

#### 首次训练

首次训练流程比较直接：加载数据 -> 特征工程 -> 训练模型 -> 保存模型。

**代码示例 (`main.py` 简化流程):**
```python
# 在 main.py 中
def train_pipeline():
    # 1. 生成数据
    df_train = generate_fake_data(1000)
    
    # 2. 特征处理 & 构建词表
    processor = FeatureProcessor(CONFIG)
    processor.fit(df_train)
    s_dict, d_tensor, t_tensor, l_tensor = processor.transform(df_train)
    
    # 3. 封装 DataLoader
    dataset = RepairDataset(s_dict, d_tensor, t_tensor, l_tensor)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    # 4. 初始化模型、优化器
    model = DeepFM(CONFIG)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    
    # 5. 训练循环
    model.train()
    for epoch in range(CONFIG["epochs"]):
        for batch in dataloader:
            # ... 训练逻辑 ...
            pass
            
    # 6. 保存模型和词表 (词表在 processor.fit 时已自动保存)
    torch.save(model.state_dict(), "deepfm_initial.pth")

# 执行
train_pipeline()
```

#### 增量训练

增量训练是项目的核心亮点，它包含了一套完整的自动化决策流程，以保证模型的稳定性和效果。

**核心逻辑 (`incre_train.py`):**
1.  **加载当天数据**，并切分为训练集和验证集。
2.  **回放历史验证集**: 从过去N天的验证集中采样一部分，与当天的验证集合并，构成最终的验证集。这可以有效防止模型在学习新知识时遗忘旧知识。
3.  **检查积压数据**: 检查是否存在上次训练效果不佳而“积压”下来的数据，如果存在，则与当天的训练集合并。
4.  **加载线上模型**: 加载当前正在服役的模型作为基准 (Baseline)。
5.  **评估基准**: 在最终的验证集上评估基准模型的表现 (e.g., AUC)。
6.  **增量训练**: 在最终的训练集上对模型进行小步长 (low learning rate) 的训练。
7.  **评估新模型**: 在同一验证集上评估新模型的表现。
8.  **决策**:
    -   如果 `New_AUC >= Old_AUC`，说明训练有效。此时，**上线新模型**，并**清空积压数据**。
    -   如果 `New_AUC < Old_AUC`，说明此次训练可能引入了噪声或导致过拟合。此时，**保持原模型不变**，并将本次使用的**训练数据存入“积压区”**，留待第二天与新的数据合并再次训练。

**代码示例 (`incre_train.py`):**
```python
# 在 incre_train.py 中
class DailyIncrementalPipeline:
    def run_daily_job(self, today_date_str, daily_df):
        # ... (数据准备、回放、检查积压) ...
        
        # 加载旧模型并评估
        model = DeepFM(self.config)
        best_model_path = self.get_current_best_model_path()
        model.load_state_dict(torch.load(best_model_path))
        old_auc, _ = self.evaluate(model, val_loader)
        
        # 增量训练
        # ... (训练循环) ...
        
        # 评估新模型
        new_auc, _ = self.evaluate(model, val_loader)
        
        # 决策
        if new_auc >= old_auc:
            print("效果提升，上线新模型！")
            # 保存模型、更新线上配置、清空积压
        else:
            print("效果未提升，回滚并积压数据。")
            # 保存训练数据到 buffer 文件
```

### 4. 线上推理 & 动态特征

线上推理时，最大的挑战是处理训练时未见过的特征值（例如新注册的品牌 "小米汽车"）。本方案通过 `Redis` 实现了一个分布式的动态词表。

**核心逻辑 (`redis_cache/online_vocab.py`):**
1.  **`OnlineVocabHandler`**: 线上服务初始化一个处理器，连接到 Redis。
2.  **Redis 数据结构**:
    -   一个 `HASH` (`feature:brand:map`) 存储 `品牌名 -> ID` 的映射。
    -   一个 `STRING` (`feature:brand:next_id`) 作为原子计数器，记录下一个可用的 ID。
3.  **查询流程**:
    -   当一个请求进来 (e.g., `brand_name="小米汽车"`)，首先在 Redis HASH 中查询。
    -   如果**命中**，直接返回 ID。
    -   如果**未命中**：
        a.  使用 `INCR` 命令对计数器原子 +1，获取一个**全局唯一**的新 ID。
        b.  检查新 ID 是否超过了模型 Embedding 层的容量限制。
        c.  如果**未超限**，使用 `HSETNX` (set if not exists) 将 `"小米汽车" -> new_id` 写入 HASH。`HSETNX` 保证了即使有多个服务实例并发处理同一个新品牌，也只有一个能写入成功，避免了竞争。
        d.  如果**已超限**，则将计数器原子 -1 (回滚)，并返回 `UNK` (未知) 对应的 ID。模型需要对 `UNK` ID 有合理的处理。

**代码示例 (`inference.py`):**
```python
# 在 inference.py 中

# 全局初始化 (服务启动时执行一次)
vocab_handler = OnlineVocabHandler(model_vocab_limit=500) 
model = DeepFM(...) 
model.load_state_dict(torch.load("model.pth"))
model.eval()

def predict(request_json):
    """处理单个线上请求"""
    brand_name = request_json.get("car_brand")
    
    # 核心：自动处理新旧品牌，获取 ID
    brand_id = vocab_handler.get_brand_id(brand_name)
    
    # 构造 Tensor
    sparse_tensor = torch.tensor([[brand_id]], dtype=torch.long)
    
    # ... (处理其他特征) ...
    
    # 推理
    with torch.no_grad():
        score = model(dense_tensor, sparse_tensor, bge_tensor)
        
    return {"score": score.item(), "matched_brand_id": brand_id}

# 模拟调用
request = {"car_brand": "小米汽车", ...}
result = predict(request)
print(result)
```

通过这套机制，系统实现了无需停机、无需重新训练模型，即可实时接纳新特征的能力，保证了业务的连续性。当新特征积累到一定程度后，离线的增量训练会自动学习这些新特征的表达。
