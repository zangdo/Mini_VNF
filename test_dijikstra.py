import os
import json
import numpy as np
import networkx as nx
from itertools import islice
from config import Config
from env import QoSRoutingEnv
from generate_request import generate_single_request

def solve_yens(env, src, dst, bw_req, max_delay, k=7):
    """
    Thuật toán Yen's K-Shortest Paths kết hợp Load Balancing.
    Tìm K đường ngắn nhất, sau đó chọn đường có Bottleneck Bandwidth to nhất.
    """
    bw_matrix = env.current_bw_matrix
    delay_matrix = env.static_delay_matrix
    base_mask = env.base_topology_mask
    num_nodes = Config.NUM_NODES

    G = nx.DiGraph()
    # Ép tạo đủ các đỉnh để tránh lỗi NodeNotFound
    G.add_nodes_from(range(num_nodes))

    # Xây dựng đồ thị (Chỉ lấy các cạnh thỏa mãn Băng thông)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if base_mask[i, j] == 1.0 and bw_matrix[i, j] >= bw_req:
                G.add_edge(i, j, weight=delay_matrix[i, j])

    try:
        # nx.shortest_simple_paths chính là thuật toán Yen's (trả về Generator)
        k_paths_generator = nx.shortest_simple_paths(G, source=src, target=dst, weight='weight')
        
        best_path = None
        best_bottleneck = -1.0

        # Rút ra tối đa K đường từ Generator
        for path in islice(k_paths_generator, k):
            # Tính tổng Delay của đường này
            path_delay = sum(delay_matrix[path[i], path[i+1]] for i in range(len(path)-1))
            
            # Yen's sắp xếp đường đi theo Delay tăng dần.
            if path_delay <= max_delay:
                # Tìm Cổ chai Băng thông (Bottleneck) của đường này
                bottleneck = min(bw_matrix[path[i], path[i+1]] for i in range(len(path)-1))
                
                # Chiến lược chọn: Đường nào có cổ chai to nhất thì đi (Load Balancing)
                if bottleneck > best_bottleneck:
                    best_bottleneck = bottleneck
                    best_path = path
            else:
                # Nếu đường thứ n đã vi phạm Max Delay, thì n+1, n+2 chắc chắn vi phạm!
                # Break luôn để tiết kiệm CPU
                break 

        return best_path

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None # Đứt đường, không tìm nổi kể cả 1 con đường

def evaluate_yens(env, num_episodes=10, k=7):
    """ Phòng thi dành riêng cho Yen's K-Shortest Path """
    print(f"\n--- BẮT ĐẦU BÀI THI YEN'S (K={k}) ({num_episodes} PHIÊN) ---")
    total_acc_rates = []
    
    for ep in range(num_episodes):
        env.reset()
        fail_count = 0
        ep_total_req = 0
        ep_success_req = 0
        
        while fail_count < Config.MAX_FAILURES:
            req = generate_single_request(Config.NUM_NODES, Config.BW_MIN, Config.BW_MAX, Config.DELAY_MIN, Config.DELAY_MAX)
            
            src = req['src']
            dst = req['dst']
            bw_req = req['bw_req']
            max_delay = req['max_delay']
            
            _ = env.setup_request(req)
            ep_total_req += 1
            
            # Gọi bộ não Yen's
            path = solve_yens(env, src, dst, bw_req, max_delay, k)
            
            if path is None:
                fail_count += 1
            else:
                is_success_req = False
                
                # Đi theo lộ trình Yen's đã vạch ra
                for next_node in path[1:]:
                    _, _, done, info = env.step(next_node)
                    
                    if done:
                        if info['status'] == 'Success':
                            is_success_req = True
                            fail_count = 0 
                        else:
                            fail_count += 1
                        break
                        
                if is_success_req:
                    ep_success_req += 1
                    
        rate = (ep_success_req / ep_total_req * 100) if ep_total_req > 0 else 0.0
        total_acc_rates.append(rate)
        
        print(f"   [Phiên {ep+1:02d}] Đáp ứng: {ep_success_req} / {ep_total_req} requests | Tỷ lệ Accept: {rate:.2f}%")
        
    avg_acc_rate = np.mean(total_acc_rates)
    print(f"\n🎯 KẾT QUẢ CHỐT SỔ YEN'S (K={k}): Acceptance Rate trung bình = {avg_acc_rate:.2f}%\n")
    return avg_acc_rate

if __name__ == "__main__":
    with open(Config.MAP_FILE, 'r', encoding='utf-8') as f:
        topology_data = json.load(f)
        
    print("🚀 KHỞI ĐỘNG HỆ THỐNG ĐÁNH GIÁ YEN'S K-SHORTEST PATH...")
    test_env = QoSRoutingEnv(Config.NUM_NODES, topology_data)
    
    # K=7 là một con số rất đẹp, đủ để cân bằng tải mà không làm chậm thuật toán
    evaluate_yens(test_env, num_episodes=10, k=7)