import threading

import redis


class OnlineVocabHandler:
    def __init__(self, redis_host='localhost', port=6379, 
                 model_vocab_limit=500, redis_key_prefix="feature:brand"):
        # 1. 连接 Redis
        self.r = redis.Redis(host=redis_host, port=port, decode_responses=True)
        self.limit = model_vocab_limit
        self.key_map = f"{redis_key_prefix}:map"      # Hash结构: {品牌名: ID}
        self.key_next_id = f"{redis_key_prefix}:next" # String结构: int
        
        # 2. L1 本地缓存 (避免每次都查Redis)
        self.local_cache = {}
        self.lock = threading.Lock() # 简单的线程锁
        
        # 3. 初始化: 首次启动时，把 Redis 里的热数据加载到本地
        self._warmup_local_cache()

    def _warmup_local_cache(self):
        """预热：把 Redis 里的映射拉取到本地内存"""
        all_data = self.r.hgetall(self.key_map)
        if all_data:
            # Redis 返回的 value 是字符串，转成 int
            self.local_cache = {k: int(v) for k, v in all_data.items()}

    def get_brand_id(self, brand_name):
        """
        线上推理的主入口函数
        """
        if not brand_name:
            return 0 # PAD
            
        # --- A. 查 L1 本地缓存 (最快) ---
        if brand_name in self.local_cache:
            return self.local_cache[brand_name]

        # --- B. 查 L2 Redis (防止其他服务器已经创建了) ---
        # 使用 setnx 逻辑或者直接查
        remote_id = self.r.hget(self.key_map, brand_name)
        if remote_id:
            id_val = int(remote_id)
            with self.lock:
                self.local_cache[brand_name] = id_val
            return id_val

        # --- C. 创建新 ID (双重检查锁 DCL 逻辑太复杂，这里用 Redis 原子操作简化) ---
        # 这是一个新词！需要分配 ID
        return self._assign_new_id(brand_name)

    def _assign_new_id(self, brand_name):
        # 1. 检查当前 ID 计数器
        # 注意：这里有极低概率并发冲突，生产环境建议用 Lua 脚本保证原子性
        # 但对于品牌这种低频新增特征，下面的逻辑足够了
        
        current_max_id = self.r.get(self.key_next_id)
        if current_max_id is None:
            # 如果 Redis 是空的，初始化为 2 (0=PAD, 1=UNK)
            current_max_id = 1 
            self.r.set(self.key_next_id, 1)
        
        current_max_id = int(current_max_id)

        # 2. 检查是否越界 (预留坑位满了没)
        if current_max_id + 1 >= self.limit:
            # 坑位满了！必须降级处理
            print(f"[Alert] Embedding buffer full! Mapping '{brand_name}' to UNK (1).")
            # 也可以选择记录日志，提醒开发人员扩容
            return 1 # UNK

        # 3. 原子递增获取新 ID
        new_id = self.r.incr(self.key_next_id)
        
        # 4. 写入 Redis 映射
        # setnx 防止两个进程同时写入同一个品牌导致 ID 覆盖
        is_new = self.r.hsetnx(self.key_map, brand_name, new_id)
        
        if not is_new:
            # 这一瞬间别的服务器抢先创建了，那我就用它的
            new_id = int(self.r.hget(self.key_map, brand_name))
        
        # 5. 更新本地缓存
        with self.lock:
            self.local_cache[brand_name] = new_id
            
        print(f"[New Brand] Assigned ID {new_id} to '{brand_name}'")
        return new_id
