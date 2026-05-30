import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def visualize_network(map_file):
    print(f"🔍 Đang phân tích bản đồ: {map_file}...")
    
    if not os.path.exists(map_file):
        print(f"❌ Không tìm thấy file {map_file}! Hãy chạy sinh map trước.")
        return

    # 1. Đọc dữ liệu từ file JSON
    with open(map_file, 'r', encoding='utf-8') as f:
        topology_data = json.load(f)

    # 2. Khởi tạo đồ thị
    G = nx.Graph()
    
    links = topology_data['links']
    bws = topology_data['bw']
    
    for (u, v), bw in zip(links, bws):
        G.add_edge(u, v, weight=bw)

    # 3. Tính toán kích thước Node (Trạm nào nhiều cáp nối -> Trạm to)
    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 20 for n in G.nodes()]

    # 4. Tính toán độ dày và màu sắc của Cạnh (Cáp to -> Dày & Đậm màu)
    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for u, v in edges]
    
    # Chia dải màu từ Nhạt (Băng thông thấp) đến Đậm (Băng thông cao)
    cmap = plt.cm.plasma 
    norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    edge_colors = [cmap(norm(w)) for w in edge_weights]
    
    # 5. Chọn thuật toán dàn trang (Layout)
    # Kamada-Kawai layout thường cho ra hình ảnh mạng viễn thông đẹp và bung đều nhất
    print("⚙️ Đang tính toán tọa độ các trạm (có thể mất vài giây)...")
    pos = nx.kamada_kawai_layout(G)

    # 6. Bắt đầu vẽ
    plt.figure(figsize=(14, 10))
    plt.title(f"Hạ tầng mạng Viễn thông - {topology_data['metadata']['num_nodes']} Nodes", fontsize=16, fontweight='bold')
    
    # Vẽ các đường cáp (Edges)
    nx.draw_networkx_edges(
        G, pos, 
        width=1.5, 
        edge_color=edge_colors, 
        alpha=0.7
    )
    
    # Vẽ các Trạm (Nodes)
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        node_color='cyan', 
        edgecolors='black', 
        linewidths=1.2
    )
    
    # Đánh số thứ tự các Trạm
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")

    # Thêm thanh chú thích màu sắc cho Băng thông (Colorbar)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label('Băng thông (Capacity)', rotation=270, labelpad=15, fontweight='bold')

    plt.axis('off')
    
    # Tự động lưu ra ảnh chất lượng cao
    output_img = map_file.replace('.json', '.png')
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu ảnh bản đồ chất lượng cao (300dpi) tại: {output_img}")
    
    # Hiển thị lên màn hình
    plt.show()

if __name__ == "__main__":
    from config import Config
    # Trực tiếp đọc đường dẫn MAP_FILE từ file config của Tú
    visualize_network(Config.MAP_FILE)