# test_DFS.py
import os
import json
import numpy as np
import tensorflow as tf
from config import Config
from env import QoSRoutingEnv
from model import RoutingPACModel
from generate_request import generate_single_request

# ==========================================
# 1. ĐỊNH NGHĨA AGENT DFS (nếu chưa có trong agent.py)
# ==========================================
class DFSRoutingAgent:
    """
    Agent sử dụng policy (PPO) để ưu tiên các node, kết hợp DFS để tìm đường
    mà không đi lại các đỉnh đã thăm. Khi gặp deadend, nó tự động quay lui.
    """
    def __init__(self, model, max_depth=30):
        self.model = model
        self.max_depth = max_depth
        self.best_path = None
        self.success = False

    def _dfs(self, env, node, dst, path, visited, remain_time):
        """
        Đệ quy tìm đường theo DFS, ưu tiên các node có xác suất cao từ policy.
        """
        if node == dst:
            self.best_path = path.copy()
            self.success = True
            return True

        if len(path) >= self.max_depth:
            return False

        # Lấy state hiện tại và action probabilities
        state = env._get_state_dict()
        action_probs, _, _ = self.model(state)
        valid_mask = state['valid_mask'].numpy()[0]
        probs = action_probs.numpy()[0]

        # Xây danh sách các node hợp lệ, chưa visited, sắp xếp theo prob giảm dần
        candidates = []
        for n in range(env.num_nodes):
            if valid_mask[n] > 0 and n not in visited:
                candidates.append((n, probs[n]))
        candidates.sort(key=lambda x: x[1], reverse=True)

        for next_node, _ in candidates:
            # Kiểm tra thời gian còn lại
            edge_delay = env.static_delay_matrix[node, next_node]
            if remain_time < edge_delay:
                continue  # không đủ thời gian, bỏ qua nhánh này

            # --- Tạm chiếm dụng băng thông ---
            env.current_bw_matrix[node, next_node] -= env.bw_req
            env.current_bw_matrix[next_node, node] -= env.bw_req

            # Đệ quy xuống nhánh
            new_visited = visited | {next_node}
            new_path = path + [next_node]
            found = self._dfs(env, next_node, dst, new_path, new_visited,
                              remain_time - edge_delay)

            # Nếu tìm thấy, dừng ngay
            if found:
                return True

            # --- Rollback băng thông nếu nhánh thất bại ---
            self._rollback_edge(env, node, next_node)

        return False

    def _rollback_edge(self, env, u, v):
        """Hoàn trả băng thông cho một cạnh cụ thể (dùng khi backtrack)"""
        env.current_bw_matrix[u, v] += env.bw_req
        env.current_bw_matrix[v, u] += env.bw_req
        # Xóa cạnh khỏi danh sách touched_edges nếu có (trong env gốc)
        if hasattr(env, 'touched_edges'):
            if (u, v) in env.touched_edges:
                env.touched_edges.remove((u, v))
            elif (v, u) in env.touched_edges:
                env.touched_edges.remove((v, u))

    def find_path(self, env):
        """
        Tìm đường từ src đến dst của env hiện tại.
        Trả về (path, status) với path là list các node hoặc None.
        """
        self.best_path = None
        self.success = False

        # Reset môi trường và lấy thông tin
        env.reset()
        src = env.src
        dst = env.dst
        remain_time = env.max_delay  # được set trong setup_request

        # Bắt đầu DFS từ src
        self._dfs(env, src, dst, [src], {src}, remain_time)

        if self.success:
            return self.best_path, 'Success'
        else:
            return None, 'Failed'


# ==========================================
# 2. HÀM ĐÁNH GIÁ
# ==========================================
def evaluate_dfs_agent(agent, env, num_episodes=20):
    """Đánh giá agent DFS với num_episodes request"""
    success_count = 0
    total_requests = 0
    print(f"\n=== ĐÁNH GIÁ AGENT DFS TRÊN {num_episodes} REQUEST ===\n")
    for ep in range(num_episodes):
        env.reset()
        req = generate_single_request(Config.NUM_NODES, Config.BW_MIN, Config.BW_MAX,
                                      Config.DELAY_MIN, Config.DELAY_MAX)
        env.setup_request(req)
        path, status = agent.find_path(env)
        total_requests += 1
        if status == 'Success':
            success_count += 1
            print(f"[{ep+1:03d}] ✅ Thành công: {req['src']}->{req['dst']} | Path: {path}")
        else:
            print(f"[{ep+1:03d}] ❌ Thất bại: {req['src']}->{req['dst']} | Path: {path if path else 'None'}")

    acc_rate = (success_count / total_requests) * 100
    print(f"\n🎯 KẾT QUẢ: Acceptance Rate = {acc_rate:.2f}% ({success_count}/{total_requests})")
    return acc_rate


# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    # --- Cấu hình đường dẫn model ---
    # Lấy tên map từ Config
    map_name = os.path.splitext(os.path.basename(Config.MAP_FILE))[0]
    model_dir = os.path.join("model_history", map_name)
    # Tìm file weights mới nhất (có thể sửa thành file cụ thể)
    import glob
    import re
    saved_models = glob.glob(os.path.join(model_dir, "model_update_*.weights.h5"))
    if saved_models:
        def extract_step(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r"update_(\d+)", filename)
            return int(match.group(1)) if match else -1
        latest_model_path = max(saved_models, key=extract_step)
        print(f"📂 Sử dụng model mới nhất: {latest_model_path}")
    else:
        print("⚠️ Không tìm thấy model đã train. Thoát.")
        exit(1)

    # --- Load topology ---
    with open(Config.MAP_FILE, 'r', encoding='utf-8') as f:
        topology_data = json.load(f)

    # --- Tạo môi trường test đơn ---
    test_env = QoSRoutingEnv(Config.NUM_NODES, topology_data)

    # --- Khởi tạo model và load weights ---
    model = RoutingPACModel(Config.NUM_NODES, Config.EMBED_DIM, Config.NUM_GCN_LAYERS)

    # Build model bằng dummy state
    dummy_req = generate_single_request(Config.NUM_NODES, Config.BW_MIN, Config.BW_MAX,
                                        Config.DELAY_MIN, Config.DELAY_MAX)
    dummy_state = test_env.setup_request(dummy_req)
    _ = model(dummy_state)

    model.load_weights(latest_model_path)
    print("✅ Đã load trọng số model.")

    # --- Tạo agent DFS ---
    dfs_agent = DFSRoutingAgent(model, max_depth=30)

    # --- Đánh giá ---
    acc = evaluate_dfs_agent(dfs_agent, test_env, num_episodes=20)