import os
import json
import random
import networkx as nx
import argparse

def generate_telecom_map(num_nodes, num_edges, bw_min, bw_max, delay_min, delay_max, output_file):
    """
    Sinh đồ thị mạng viễn thông mô phỏng thực tế.
    """
    if num_edges < num_nodes - 1:
        raise ValueError(f"Lỗi: Số cạnh ({num_edges}) phải >= Số node - 1 ({num_nodes - 1}) để mạng liên thông!")

    print(f"⚙️ Bắt đầu sinh bản đồ: {num_nodes} Nodes, {num_edges} Edges...")

    # BƯỚC 1: Tạo bộ khung xương (Spanning Tree) để đảm bảo không có Node nào bị cô lập
    G = nx.random_labeled_tree(n=num_nodes)

    # BƯỚC 2: Thêm các cạnh còn lại theo cơ chế "Ưu tiên Trạm Lớn" (Preferential Attachment)
    edges_to_add = num_edges - (num_nodes - 1)
    nodes_list = list(G.nodes())

    while edges_to_add > 0:
        # Lấy bậc (degree) của các node làm trọng số. Node nào nhiều cáp sẽ dễ được chọn thêm.
        degrees = [G.degree(n) for n in nodes_list]
        
        # Chọn ngẫu nhiên 2 node dựa trên trọng số bậc
        u, v = random.choices(nodes_list, weights=degrees, k=2)
        
        # Đảm bảo không tạo vòng lặp (self-loop) và không nối đè cạnh đã có
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            edges_to_add -= 1

    # BƯỚC 3: Gán ngẫu nhiên tài nguyên (Băng thông và Độ trễ)
    links = []
    bw_list = []
    delay_list = []

    for u, v in G.edges():
        links.append([u, v])
        # Làm tròn Băng thông đến 1 chữ số thập phân (Ví dụ: 50.5 Gbps)
        bw = round(random.uniform(bw_min, bw_max), 1)
        # Làm tròn Độ trễ đến 2 chữ số thập phân (Ví dụ: 3.14 ms)
        delay = round(random.uniform(delay_min, delay_max), 2)
        
        bw_list.append(bw)
        delay_list.append(delay)

    # BƯỚC 4: Đóng gói dữ liệu chuẩn Form của env.py
    topology_data = {
        "metadata": {
            "description": "Generated Realistic Telecom Network",
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "bw_range": [bw_min, bw_max],
            "delay_range": [delay_min, delay_max]
        },
        "links": links,
        "bw": bw_list,
        "delay": delay_list
    }

    # Tạo folder chứa map nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Xuất file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(topology_data, f, indent=4)
        
    print(f"✅ Đã lưu thành công tại: {output_file}")


if __name__ == "__main__":
    # Thiết lập bộ đọc tham số dòng lệnh (Command-line arguments)
    parser = argparse.ArgumentParser(description="Trình tạo bản đồ mạng NFV DRL")
    parser.add_argument("--nodes", type=int, default=100, help="Số lượng Node (Mặc định: 100)")
    parser.add_argument("--edges", type=int, default=200, help="Số lượng Cạnh (Mặc định: 200)")
    parser.add_argument("--bw_min", type=float, default=10.0, help="Băng thông Min")
    parser.add_argument("--bw_max", type=float, default=100.0, help="Băng thông Max")
    parser.add_argument("--delay_min", type=float, default=1.0, help="Độ trễ Min (ms)")
    parser.add_argument("--delay_max", type=float, default=30.0, help="Độ trễ Max (ms)")
    parser.add_argument("--name", type=str, default="map_100_200.json", help="Tên file JSON đầu ra")

    args = parser.parse_args()

    filepath = os.path.join("map", args.name)
    
    generate_telecom_map(
        num_nodes=args.nodes,
        num_edges=args.edges,
        bw_min=args.bw_min,
        bw_max=args.bw_max,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        output_file=filepath
    )