import matplotlib
matplotlib.use('Agg')
import os
import json
import glob
import re
import numpy as np
import tensorflow as tf
from config import Config
from env import BatchedGpuQoSRoutingEnv, QoSRoutingEnv # Import cả 2 env
from model import RoutingPACModel
from agent import BatchedPPOAgent
from generate_request import generate_batched_requests_gpu, generate_single_request

# ==========================================
# CÀI ĐẶT THƯ MỤC CHECKPOINT
# ==========================================
map_name = os.path.splitext(os.path.basename(Config.MAP_FILE))[0]
model_dir = os.path.join("model_history", map_name)
os.makedirs(model_dir, exist_ok=True)

with open(Config.MAP_FILE, 'r', encoding='utf-8') as f:
    topology_data = json.load(f)
with tf.device('/GPU:0'):
    # 1. Khởi tạo Môi trường song song trên GPU cho TRAIN
    train_env = BatchedGpuQoSRoutingEnv(Config.NUM_NODES, topology_data, batch_size=Config.BATCH_SIZE)
    model = RoutingPACModel(Config.NUM_NODES, Config.EMBED_DIM, Config.NUM_GCN_LAYERS)
    agent = BatchedPPOAgent(model, Config.LR, Config.GAMMA, Config.LAMBDA, Config.CLIP_RATIO)
# 2. Khởi tạo Môi trường đơn truyền thống cho TEST (Phòng thi)
test_env = QoSRoutingEnv(Config.NUM_NODES, topology_data)
print("🚀 KHỞI ĐỘNG HỆ THỐNG END-TO-END GPU DRL TRÊN A100...")

# ==========================================
# AUTO-RESUME: TÌM VÀ LOAD MODEL MỚI NHẤT
# ==========================================
update_step = 0

# Cú lừa Build Model: Dùng hàm generate request mới cho GPU
with tf.device('/GPU:0'):
    dummy_reqs = generate_batched_requests_gpu(Config.BATCH_SIZE, Config.NUM_NODES, Config.BW_MIN, Config.BW_MAX, Config.DELAY_MIN, Config.DELAY_MAX)
    dummy_state = train_env.setup_requests(dummy_reqs['src'], dummy_reqs['dst'], dummy_reqs['bw_req'], dummy_reqs['max_delay'])
    _ = agent.model(dummy_state) 

saved_models = glob.glob(os.path.join(model_dir, "model_update_*.weights.h5"))

if saved_models:
    def extract_step(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r"update_(\d+)", filename)
        return int(match.group(1)) if match else -1

    latest_model_path = max(saved_models, key=extract_step)
    update_step = extract_step(latest_model_path)
    
    agent.model.load_weights(latest_model_path)
    print(f"🔄 ĐÃ TÌM THẤY NÃO CŨ! Nạp thành công trọng số từ: {latest_model_path}")
    print(f"📈 Sẽ tiếp tục Train từ mẻ thứ {update_step + 1}...")
else:
    print("✨ Bắt đầu Train một bộ não hoàn toàn mới (From Scratch)...")

# ==========================================
# HÀM ĐÁNH GIÁ (PHÒNG THI ĐỘC LẬP)
# ==========================================
def evaluate_model(agent, env, num_episodes=10):
    """ 
    Kiểm tra độc lập. Tắt nhiễu. Đi tới khi mạng sập (đủ MAX_FAILURES).
    In chi tiết từng phiên và trả về Trung bình Acceptance Rate.
    """
    total_acc_rates = []
    print(f"\n--- BẮT ĐẦU BÀI THI ({num_episodes} PHIÊN) ---")
    
    for ep in range(num_episodes):
        env.reset()
        fail_count = 0
        ep_total_req = 0
        ep_success_req = 0
        
        while fail_count < Config.MAX_FAILURES:
            req = generate_single_request(Config.NUM_NODES, Config.BW_MIN, Config.BW_MAX, Config.DELAY_MIN, Config.DELAY_MAX)
            state = env.setup_request(req)
            
            # Khởi đầu đã không có đường
            if np.sum(state['valid_mask'].numpy()[0]) == 0.0:
                fail_count += 1
                ep_total_req += 1
                continue
                
            done = False
            ep_total_req += 1
            is_success_req = False
            
            while not done:
                # Lấy trực tiếp action có prob cao nhất (Argmax) để test (TẮT NHIỄU)
                action_probs, _, _ = agent.model(state)
                action_val = tf.argmax(action_probs, axis=-1).numpy()[0]
                
                next_state, _, done, info = env.step(action_val)
                state = next_state
                
                if done:
                    if info['status'] == 'Success':
                        is_success_req = True
                        fail_count = 0 # Reset fail count nếu cứu sống được mạng
                    else: # DeadEnd
                        fail_count += 1
                        
            if is_success_req:
                ep_success_req += 1
                
        # Tính tỷ lệ cho phiên hiện tại
        rate = (ep_success_req / ep_total_req * 100) if ep_total_req > 0 else 0.0
        total_acc_rates.append(rate)
        
        # IN RA KẾT QUẢ NGAY LẬP TỨC CHO PHIÊN NÀY
        print(f"   [Phiên {ep+1:02d}] Đáp ứng: {ep_success_req} / {ep_total_req} requests | Tỷ lệ Accept: {rate:.2f}%")
        
    # Lấy trung bình phần trăm của tất cả các phiên
    avg_acc_rate = np.mean(total_acc_rates)
    return avg_acc_rate

# ==========================================
# VÒNG LẶP HUẤN LUYỆN CHÍNH TRÊN GPU
# ==========================================
while update_step < Config.NUM_EPOCHS:
    update_step += 1
    print(f"🔥 [Update {update_step}] Đang thả 1024 môi trường cày cuốc trên GPU...")
    
    # 1. TRAIN BATCHED PPO TRÊN GPU (Gồm cả Rollout và Train cực nhanh)
    # 1 Epoch lớn này tương đương thu thập 131,072 Transitions!
    loss_metrics = agent.learn(train_env, num_steps=Config.NUM_STEPS, ppo_epochs=Config.PPO_EPOCHS, minibatch_size=Config.MINIBATCH_SIZE)
    
    print(f"✅ Train xong! A-Loss: {loss_metrics['actor_loss']:.3f} | C-Loss: {loss_metrics['critic_loss']:.3f} | Entropy: {loss_metrics['entropy']:.3f}")
    
    # 2. KIỂM TRA ĐỊNH KỲ BẰNG MÔI TRƯỜNG ĐƠN
    if update_step % Config.TEST_PER_UPDATE_STEP == 0:
        print("⏳ Đang làm bài thi đánh giá năng lực...")
        # Ép thi 10 phiên theo ý Tú (hoặc dùng Config.NUM_EPISODES_TEST)
        test_acc_rate = evaluate_model(agent, test_env, num_episodes=Config.NUM_EPISODES_TEST) 
        # Chỉ in đúng cái phần trăm trung bình ở cuối cùng
        print(f"🎯 KẾT QUẢ CHỐT SỔ: Acceptance Rate trung bình = {test_acc_rate:.2f}%\n")
    
    # 3. LƯU MODEL MỖI 100 UPDATE
    if update_step % Config.MODEL_SAVE_PER_UPDATE_STEP == 0:
        save_path = os.path.join(model_dir, f"model_update_{update_step}.weights.h5")
        agent.model.save_weights(save_path)
        print(f"💾 ĐÃ LƯU CHECKPOINT AN TOÀN VÀO: {save_path}\n")