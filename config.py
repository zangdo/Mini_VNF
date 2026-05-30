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
    ROLLOUT_SIZE = 2048    # Số data gom trước khi Train
    MINIBATCH_SIZE = 256   # Kích thước 1 mẻ đưa vào GPU
    PPO_EPOCHS = 4         # Số lần nhai lại 1 mẻ data
    LR = 3e-4
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_RATIO = 0.2
    
    # --- ĐIỀU KIỆN MÔI TRƯỜNG ---
    MAX_FAILURES = 10      # Số lần chết liên tục trước khi Reset Map