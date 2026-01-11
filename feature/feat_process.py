import json
from typing import Optional

# 假设车品牌定义时预留了 500 个坑位
CAR_BRAND_VOCAB_SIZE_LIMIT = 500 

class BrandVocabManager:
    def __init__(self, initial_brands_list=None):
        # 定义特殊 Token
        self.PAD_TOKEN = "<PAD>" # ID 0
        self.UNK_TOKEN = "<UNK>" # ID 1 (遇到真·未知的兜底)
        
        # 内存中的映射表
        self.token2id = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.id2token = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        
        # 记录当前用到了哪个ID
        self.next_id = 2
        
        # 如果有初始列表，直接构建
        if initial_brands_list:
            for brand in initial_brands_list:
                self.add_brand(brand)


    def add_brand(self, brand_name):
        if brand_name not in self.token2id:
            self.token2id[brand_name] = self.next_id
            self.id2token[self.next_id] = brand_name
            self.next_id += 1
            return self.next_id
        return self.token2id[brand_name]

    def get_id(self, brand_name):
        # 查不到就返回 UNK (1)
        return self.token2id.get(brand_name, 1)
    
    def save_to_json(self,file_path:Optional[str]=None):
        """保存到本地文件，用于版本控制或离线训练"""
        if not file_path:
            print("请输入需要保存的文件路径 ")
            return 

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "token2id": self.token2id,
                "next_id": self.next_id
            }, f, ensure_ascii=False, indent=2)

    def load_from_json(self,file_path:Optional[str]=None):

        if not file_path:
            print("请输入需要打开的文件路径 ")
            return 
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.token2id = data["token2id"]
            # 反转生成 id2token
            self.id2token = {v: k for k, v in self.token2id.items()}
            self.next_id = data["next_id"]

    def sync_to_redis(self, redis_host='localhost', port=6379):
        """同步到 Redis，供线上 Java/Go 服务直接读取"""
        r = redis.Redis(host=redis_host, port=port, decode_responses=True)
        # 使用 Hash 结构存储
        r.hmset("feature:brand:map", self.token2id)
        r.set("feature:brand:next_id", self.next_id)



def handle_new_brand_request(manager, new_brand_name):
    """
    处理线上新出现的品牌
    """
    # 1. 检查是否已存在
    if new_brand_name in manager.token2id:
        return manager.get_id(new_brand_name)
    
    # 2. 检查是否有空余坑位
    if manager.next_id >= CAR_BRAND_VOCAB_SIZE_LIMIT:
        print(f"[警告] 词表已满 ({manager.next_id})！新品牌 {new_brand_name} 暂时映射为 UNK，请安排模型扩容重训。")
        return 1 # 返回 UNK
    
    # 3. 分配新ID并保存
    new_id = manager.add_brand(new_brand_name)
    manager.save_to_json() # 立即持久化
    # manager.sync_to_redis() # 同步线上
    
    print(f"[成功] 新品牌 '{new_brand_name}' 已入库，分配 ID: {new_id}。模型 Embedding 第 {new_id} 行参数目前是随机初始化的。")
    return new_id



