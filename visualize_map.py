import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def visualize_network(map_file):
    print(f"🔍 Đang phân tích bản đồ: {map_file}...")
    
    if not os.path.exists(map_file):
        print(f"Không tìm thấy file {map_file}")
        return
    with open(map_file, 'r', encoding='utf-8') as f:
        topology_data = json.load(f)
    G = nx.Graph()
    
    links = topology_data['links']
    bws = topology_data['bw']
    
    for (u, v), bw in zip(links, bws):
        G.add_edge(u, v, weight=bw)
    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 20 for n in G.nodes()]
    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for u, v in edges]
    cmap = plt.cm.plasma 
    norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    edge_colors = [cmap(norm(w)) for w in edge_weights]
    print("⚙️ Đang tính toán tọa độ các trạm (có thể mất vài giây)...")
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(14, 10))
    plt.title(f"Hạ tầng mạng Viễn thông - {topology_data['metadata']['num_nodes']} Nodes", fontsize=16, fontweight='bold')
    nx.draw_networkx_edges(
        G, pos, 
        width=1.5, 
        edge_color=edge_colors, 
        alpha=0.7
    )
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        node_color='cyan', 
        edgecolors='black', 
        linewidths=1.2
    )
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label('Băng thông (Capacity)', rotation=270, labelpad=15, fontweight='bold')

    plt.axis('off')
    output_img = map_file.replace('.json', '.png')
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu ảnh bản đồ chất lượng cao (300dpi) tại: {output_img}")
    plt.show()

if __name__ == "__main__":
    from config import Config
    visualize_network(Config.MAP_FILE)