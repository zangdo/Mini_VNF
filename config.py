# config.py
class Config:
    # --- THÔNG SỐ BẢN ĐỒ & REQUEST ---
    MAP_FILE = "map/map_30_dense.json"
    NUM_NODES = 30
    BW_MIN = 10.0
    BW_MAX = 50.0
    DELAY_MIN = 50.0
    DELAY_MAX = 100.0
    
    # --- THÔNG SỐ MODEL ---
    EMBED_DIM = 64
    NUM_GCN_LAYERS = 3
    
    # --- THÔNG SỐ PPO (A100 Tối ưu) ---
    NUM_EPOCHS = 5000
    BATCH_SIZE = 1024     # Số lượng môi trường song song trên GPU
    MINIBATCH_SIZE =  8192  # Kích thước 1 mẻ đưa vào GPU
    PPO_EPOCHS = 8         # Số lần nhai lại 1 mẻ data
    NUM_STEPS = 128       # Số bước thu thập dữ liệu trước khi cập nhật (128 x 1024 = 131,072 transitions)
    LR = 3e-4
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_RATIO = 0.2
    TEST_PER_UPDATE_STEP = 100
    MODEL_SAVE_PER_UPDATE_STEP = 100
    # --- ĐIỀU KIỆN MÔI TRƯỜNG ---
    MAX_FAILURES = 10      # Số lần chết liên tục trước khi Reset Map

    # ---TEST---
    NUM_EPISODES_TEST = 10