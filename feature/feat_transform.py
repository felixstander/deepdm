from .feat_process import BrandVocabManager, handle_new_brand_request


def main():
    # --- 模拟执行 ---
    initial_data = ["宝马", "奔驰", "奥迪", "丰田", "本田"]
    file_path = "vocab_map/brand_map.json"
    vocab_manager = BrandVocabManager(initial_data)
    vocab_manager.save_to_json(file_path)

    print(f"当前词表大小: {len(vocab_manager.token2id)}")
    print(f"宝马 ID: {vocab_manager.get_id('宝马')}")

if __name__ == "__main__":
    main()

