import matplotlib
matplotlib.use('Agg')
import random
import json
import os
import argparse
import tensorflow as tf

def generate_single_request(num_nodes, bw_min, bw_max, delaymax_min, delaymax_max):
    """
    Sinh ra một Request định tuyến ngẫu nhiên.
    Đảm bảo Node Nguồn (src) và Node Đích (dst) không bao giờ trùng nhau.
    """
    src = random.randint(0, num_nodes - 1)
    dst = random.randint(0, num_nodes - 1)
    
    # Rút lại ngẫu nhiên nếu trùng đích
    while src == dst:
        dst = random.randint(0, num_nodes - 1)

    # Làm tròn thông số để dữ liệu thực tế hơn (giống log của trạm viễn thông)
    bw_req = round(random.uniform(bw_min, bw_max), 1)
    max_delay = round(random.uniform(delaymax_min, delaymax_max), 2)

    return {
        'src': src,
        'dst': dst,
        'bw_req': bw_req,
        'max_delay': max_delay
    }

def generate_batched_requests_gpu(batch_size, num_nodes, bw_min, bw_max, delaymax_min, delaymax_max, output_file=None):
    """
    Sinh hàng ngàn requests song song trực tiếp trên GPU.
    Trả về Dictionary chứa các Tensors kích thước [Batch].
    """
    # 1. Sinh Source ngẫu nhiên từ 0 đến num_nodes - 1
    src = tf.random.uniform(shape=[batch_size], minval=0, maxval=num_nodes, dtype=tf.int32)
    
    # 2. Sinh Destination ngẫu nhiên (TUYỆT KỸ TRÁNH TRÙNG LẶP SANG SOURCE)
    # Cộng thêm một độ dời (offset) ngẫu nhiên từ 1 đến num_nodes - 1, sau đó chia lấy dư.
    # Đảm bảo 100% dst luôn khác src mà không cần vòng lặp while!
    offset = tf.random.uniform(shape=[batch_size], minval=1, maxval=num_nodes, dtype=tf.int32)
    dst = (src + offset) % num_nodes
    
    # 3. Sinh Băng thông và Delay
    bw_req = tf.random.uniform(shape=[batch_size], minval=bw_min, maxval=bw_max, dtype=tf.float32)
    max_delay = tf.random.uniform(shape=[batch_size], minval=delaymax_min, maxval=delaymax_max, dtype=tf.float32)
    
    # Gom thành Tensors, sẵn sàng "bón" thẳng vào hàm setup_requests của BatchedGpuQoSRoutingEnv
    batched_requests = {
        'src': src,
        'dst': dst,
        'bw_req': bw_req,
        'max_delay': max_delay
    }

    # Tùy chọn lưu file JSON (Phục vụ việc lưu log/curriculum learning)
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Chỉ chuyển về CPU (numpy) khi cần xuất file text
        s_np, d_np = src.numpy(), dst.numpy()
        b_np, md_np = bw_req.numpy(), max_delay.numpy()
        
        reqs_list = [
            {
                'src': int(s_np[i]),
                'dst': int(d_np[i]),
                'bw_req': float(b_np[i]),
                'max_delay': float(md_np[i])
            }
            for i in range(batch_size)
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reqs_list, f, indent=4)
        print(f"✅ Đã lưu kịch bản {batch_size} requests tại: {output_file}")

    return batched_requests

if __name__ == "__main__":
    # Giao diện dòng lệnh (CLI) để test nhanh trên Terminal A100
    parser = argparse.ArgumentParser(description="Trình tạo luồng Request mạng NFV")
    parser.add_argument("--num_req", type=int, default=10, help="Số lượng Request cần sinh")
    parser.add_argument("--nodes", type=int, default=100, help="Tổng số Node của Map hiện tại")
    parser.add_argument("--bw_min", type=float, default=5.0, help="Băng thông yêu cầu Min")
    parser.add_argument("--bw_max", type=float, default=25.0, help="Băng thông yêu cầu Max")
    parser.add_argument("--delay_min", type=float, default=5.0, help="Giới hạn trễ Min")
    parser.add_argument("--delay_max", type=float, default=30.0, help="Giới hạn trễ Max")
    parser.add_argument("--save", type=str, default="", help="Đường dẫn lưu file JSON (VD: data/req_ep1.json)")

    args = parser.parse_args()

    # Sinh và In ra kết quả
    batch = generate_batched_requests_gpu(
        batch_size=args.num_req,
        num_nodes=args.nodes,
        bw_min=args.bw_min,
        bw_max=args.bw_max,
        delaymax_min=args.delay_min,
        delaymax_max=args.delay_max,
        output_file=args.save if args.save else None
    )
    
    if not args.save:
        print(json.dumps(batch, indent=4))