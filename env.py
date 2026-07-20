
import numpy as np
import tensorflow as tf

class BatchedGpuQoSRoutingEnv(tf.Module):
    def __init__(self, num_nodes, topology_data, batch_size=1024, num_failures=10):
        super().__init__()
        self.num_nodes = num_nodes
        self.B = batch_size
        self.num_failures = float(num_failures)

        # Khởi tạo ma trận vật lý dạng Numpy
        base_mask = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        delay_mat = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        cap_mat = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        for (u, v), bw, delay in zip(topology_data['links'], topology_data['bw'], topology_data['delay']):
            base_mask[u, v] = base_mask[v, u] = 1.0
            delay_mat[u, v] = delay_mat[v, u] = delay
            cap_mat[u, v] = cap_mat[v, u] = bw

        # ==========================================
        # ĐƯA STATIC TENSORS LÊN GPU VÀ NHÂN BẢN THÀNH BATCH
        # ==========================================
        self.base_mask = tf.constant(np.tile(base_mask, (self.B, 1, 1)))       # [B, N, N]
        self.static_delay = tf.constant(np.tile(delay_mat, (self.B, 1, 1)))    # [B, N, N]
        self.max_capacity = tf.constant(np.tile(cap_mat, (self.B, 1, 1)))      # [B, N, N]
        
        # Biến trạng thái động (Dynamic States) lưu trên VRAM
        self.current_bw_matrix = tf.Variable(self.max_capacity, trainable=False)
        self.snapshot_bw_matrix = tf.Variable(self.max_capacity, trainable=False) # Dùng để Rollback siêu tốc
        # SỬA Ở ĐÂY: Đổi int32 thành float32
        zero_fails = tf.zeros([self.B], dtype=tf.float32)
        self.fail_count = tf.Variable(zero_fails, trainable=False)

    def reset_all_bandwidth(self):
        """ Reset vật lý toàn bộ mạng """
        self.current_bw_matrix.assign(self.max_capacity)

    def setup_requests(self, src_b, dst_b, bw_req_b, max_delay_b):
        """ 
        Đầu vào là các Tensor shape [B]. Khởi tạo trạng thái cho B request cùng lúc.
        """
        self.src = tf.cast(src_b, tf.int32)
        self.dst = tf.cast(dst_b, tf.int32)
        self.bw_req = tf.cast(bw_req_b, tf.float32)
        self.max_delay = tf.cast(max_delay_b, tf.float32)
        
        self.curr_node = self.src
        self.remain_time = self.max_delay
        
        # Visited mask: Thay vì dùng list, ta dùng ma trận Boolean [B, N]
        self.visited = tf.cast(tf.one_hot(self.src, self.num_nodes), tf.bool)
        
        # CHỤP ẢNH BĂNG THÔNG: Lưu lại trạng thái lúc bắt đầu để Rollback
        self.snapshot_bw_matrix.assign(self.current_bw_matrix)
        
        return self._get_state_dict()

    @tf.function(jit_compile=True)
    def _symmetric_normalize(self, adj):
        """ Chuẩn hóa Đồ thị song song trên GPU """
        # adj: [B, N, N]
        adj_self = adj + tf.eye(self.num_nodes, batch_shape=[self.B])
        degree = tf.reduce_sum(adj_self, axis=-1)
        
        # Chống chia cho 0
        d_inv_sqrt = tf.where(degree > 0, tf.math.rsqrt(degree), 0.0)
        d_inv_sqrt_mat = tf.linalg.diag(d_inv_sqrt) # [B, N, N]
        
        return d_inv_sqrt_mat @ adj_self @ d_inv_sqrt_mat

    @tf.function(jit_compile=True)
    def _get_state_dict(self):
        # Expand dims để so sánh
        bw_req_exp = tf.reshape(self.bw_req, [self.B, 1, 1])
        remain_time_exp = tf.reshape(self.remain_time, [self.B, 1, 1])
        
        # Tạo mask GCN song song
        pruned_bw_mask = tf.cast(self.current_bw_matrix >= bw_req_exp, tf.float32) * self.base_mask
        pruned_delay_mask = tf.cast(self.static_delay <= remain_time_exp, tf.float32) * self.base_mask
        
        A_bw_norm = self._symmetric_normalize(pruned_bw_mask)
        A_delay_norm = self._symmetric_normalize(pruned_delay_mask)
        
        # ==========================================
        # TẠO VALID MASK (Kỹ thuật Đại số tuyến tính)
        # ==========================================
        curr_node_one_hot = tf.one_hot(self.curr_node, self.num_nodes) # [B, N]
        
        # Trích xuất hàng của curr_node từ ma trận base, bw và delay
        # Dùng phép tính batch matrix multiplication (bmm) bằng einsum cho nhanh
        curr_base = tf.einsum('bi,bij->bj', curr_node_one_hot, self.base_mask)
        curr_bw = tf.einsum('bi,bij->bj', curr_node_one_hot, self.current_bw_matrix)
        curr_delay = tf.einsum('bi,bij->bj', curr_node_one_hot, self.static_delay)
        
        bw_req_2d = tf.reshape(self.bw_req, [self.B, 1])
        remain_time_2d = tf.reshape(self.remain_time, [self.B, 1])
        
        # Logic gộp: Có nối cáp AND Đủ băng thông AND Đủ thời gian AND Chưa tới (not visited)
        valid_mask = (curr_base == 1.0) & (curr_bw >= bw_req_2d) & (curr_delay <= remain_time_2d) & (~self.visited)
        valid_mask = tf.cast(valid_mask, tf.float32)
        
        return {
            'A_bw': A_bw_norm,
            'A_delay': A_delay_norm,
            'bw_req': bw_req_2d,
            'remain_time': remain_time_2d,
            'curr_node': self.curr_node,
            'dst_node': self.dst,
            'valid_mask': valid_mask
        }

    @tf.function(jit_compile=True)
    def _calculate_jain_index(self):
        """ Tính Jain score song song cho B batch """
        utilization = (self.max_capacity - self.current_bw_matrix) / (self.max_capacity + 1e-9)
        active_mask = (self.base_mask == 1.0)
        
        # Chuyển các chỗ không có cáp thành 0
        active_util = tf.where(active_mask, utilization, 0.0)
        
        sum_util = tf.reduce_sum(active_util, axis=[1, 2])
        sum_sq_util = tf.reduce_sum(active_util**2, axis=[1, 2]) + 1e-9
        
        num_active_links = tf.reduce_sum(tf.cast(active_mask, tf.float32), axis=[1, 2])
        
        return (sum_util**2) / (num_active_links * sum_sq_util)

    @tf.function(jit_compile=True)
    def step(self, actions):
        """ actions shape: [B] """
        actions = tf.cast(actions, tf.int32)
        # 1. Trích xuất valid_mask từ state cũ (để xử phạt nếu bốc nhầm)
        old_state = self._get_state_dict()
        old_valid_mask = old_state['valid_mask']
        action_one_hot = tf.one_hot(actions, self.num_nodes) # [B, N]
        
        # Check xem action có nằm trong valid_mask không
        is_invalid_action = tf.reduce_sum(old_valid_mask * action_one_hot, axis=1) == 0.0 # [B]
        
        # ==========================================
        # 2. TẠM ỨNG TÀI NGUYÊN (Dùng Toán học One-hot, không vòng lặp)
        # ==========================================
        curr_one_hot = tf.one_hot(self.curr_node, self.num_nodes) # [B, N]
        
        # Tạo mask 2D cho cạnh [curr_node, action] và ngược lại
        edge_mask = tf.expand_dims(curr_one_hot, 2) * tf.expand_dims(action_one_hot, 1)
        edge_mask = edge_mask + tf.linalg.matrix_transpose(edge_mask) # [B, N, N]
        
        # Trừ băng thông
        bw_deduction = edge_mask * tf.reshape(self.bw_req, [self.B, 1, 1])
        new_bw = self.current_bw_matrix - bw_deduction
        self.current_bw_matrix.assign(new_bw)
        
        # Trừ thời gian delay
        edge_delays = tf.reduce_sum(self.static_delay * edge_mask, axis=[1, 2]) / 2.0 # Chia 2 vì ma trận đối xứng
        self.remain_time = self.remain_time - edge_delays
        
        # Đánh dấu Visited và Cập nhật curr_node
        self.visited = self.visited | tf.cast(action_one_hot, tf.bool)
        self.curr_node = actions
        
        # ==========================================
        # 3. KIỂM TRA PHẦN THƯỞNG VÀ ĐIỀU KIỆN DỪNG
        # ==========================================
        next_state = self._get_state_dict()
        next_valid_mask = next_state['valid_mask']
        
        is_success = (self.curr_node == self.dst)
        # SỬA LẠI THẾ NÀY: Dùng toán tử bitwise | (OR) thay cho chữ or
        is_deadend = (tf.reduce_sum(next_valid_mask, axis=1) == 0.0) | is_invalid_action
        
        # (Nếu ở đâu đó Tú có dùng and thì cũng đổi thành & nhé)
        
        dones = is_success | is_deadend
        jain_score = self._calculate_jain_index()
        
        rewards = tf.where(is_success, 3.0 + jain_score,
                  tf.where(is_deadend, -3.0 + jain_score, 0.0))
                  
        # ==========================================
        # 4. CẬP NHẬT FAIL_COUNT VÀ CHECK HARD RESET
        # ==========================================
        # Logic: Nếu DeadEnd thì +1, nếu Success thì reset về 0, nếu đang đi (In_Progress) thì giữ nguyên
        updated_fail_count = tf.where(is_deadend, self.fail_count + 1.0,
                             tf.where(is_success, 0.0, self.fail_count))
        
        # SỬA Ở ĐÂY: Dùng 10.0 thay vì 10
        needs_hard_reset = updated_fail_count >= self.num_failures
        
        # ==========================================
        # 5. XỬ LÝ ROLLBACK / HARD RESET TRÊN MATRIX
        # ==========================================
        # Bước A: Xử lý Rollback tạm thời cho ca DeadEnd thông thường
        dones_expanded = tf.reshape(is_deadend, [self.B, 1, 1])
        rolled_back_bw = tf.where(dones_expanded, self.snapshot_bw_matrix, self.current_bw_matrix)
        
        # Bước B: Xử lý đè Ma trận vật lý gốc nếu dính Hard Reset (Đủ 10 lần fail)
        reset_expanded = tf.reshape(needs_hard_reset, [self.B, 1, 1])
        final_bw = tf.where(reset_expanded, self.max_capacity, rolled_back_bw)
        self.current_bw_matrix.assign(final_bw)
        
        # Cập nhật Snapshot tương ứng
        success_expanded = tf.reshape(is_success, [self.B, 1, 1])
        new_snapshot = tf.where(success_expanded, self.current_bw_matrix, self.snapshot_bw_matrix)
        # Nếu nổ Hard reset thì snapshot cũng phải được làm sạch về max_capacity
        final_snapshot = tf.where(reset_expanded, self.max_capacity, new_snapshot)
        self.snapshot_bw_matrix.assign(final_snapshot)
        
        # Bước C: Áp giá trị fail_count mới (thằng nào vừa bị hard reset thì xé nháp về 0)
        final_fail_count = tf.where(needs_hard_reset, 0.0, updated_fail_count)
        self.fail_count.assign(final_fail_count)
        
        return next_state, rewards, dones
class QoSRoutingEnv:
    def __init__(self, num_nodes, topology_data):
        self.num_nodes = num_nodes
        self.base_topology_mask = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        self.static_delay_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        self.max_capacity_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for (u, v), bw, delay in zip(topology_data['links'], topology_data['bw'], topology_data['delay']):
            self.base_topology_mask[u, v] = 1.0
            self.base_topology_mask[v, u] = 1.0
            self.static_delay_matrix[u, v] = delay
            self.static_delay_matrix[v, u] = delay
            self.max_capacity_matrix[u, v] = bw
            self.max_capacity_matrix[v, u] = bw
        self.current_bw_matrix = self.max_capacity_matrix.copy()
    def reset(self):
        """ Khôi phục 100% băng thông như thiết kế vật lý ban đầu """
        self.current_bw_matrix = self.max_capacity_matrix.copy()
    def setup_request(self, request):
        """ Khởi tạo trạng thái cho Request mới """
        self.src = request['src']
        self.dst = request['dst']
        self.bw_req = request['bw_req']
        self.max_delay = request['max_delay']
        self.curr_node = self.src
        self.remain_time = self.max_delay
        self.visited = [self.src]
        # --- THÊM DÒNG NÀY: Lưu vết các cạnh đã tạm ứng băng thông ---
        self.touched_edges = []
        return self._get_state_dict()



    def _symmetric_normalize(self, adj):

        adj_self = adj + np.eye(self.num_nodes, dtype=np.float32)

        degree = np.sum(adj_self, axis=-1)

        d_inv_sqrt = np.power(degree, -0.5, where=degree > 0, out=np.zeros_like(degree))

        d_inv_sqrt_mat = np.diag(d_inv_sqrt)

        return d_inv_sqrt_mat @ adj_self @ d_inv_sqrt_mat



    def _get_state_dict(self):

        pruned_bw_mask = (self.current_bw_matrix >= self.bw_req) * self.base_topology_mask

        pruned_delay_mask = (self.static_delay_matrix <= self.remain_time) * self.base_topology_mask

       

        A_bw_norm = self._symmetric_normalize(pruned_bw_mask)

        A_delay_norm = self._symmetric_normalize(pruned_delay_mask)

       

        valid_mask = np.zeros(self.num_nodes, dtype=np.float32)

        for neighbor in range(self.num_nodes):

            if (self.base_topology_mask[self.curr_node, neighbor] == 1.0 and

                self.current_bw_matrix[self.curr_node, neighbor] >= self.bw_req and

                self.static_delay_matrix[self.curr_node, neighbor] <= self.remain_time and

                neighbor not in self.visited):

                valid_mask[neighbor] = 1.0

               

        return {

            'A_bw': tf.convert_to_tensor(np.expand_dims(A_bw_norm, axis=0), dtype=tf.float32),
            'A_delay': tf.convert_to_tensor(np.expand_dims(A_delay_norm, axis=0), dtype=tf.float32),
            'bw_req': tf.convert_to_tensor([[self.bw_req]], dtype=tf.float32),
            'remain_time': tf.convert_to_tensor([[self.remain_time]], dtype=tf.float32),
            'curr_node': tf.convert_to_tensor([self.curr_node], dtype=tf.int32),
            'dst_node': tf.convert_to_tensor([self.dst], dtype=tf.int32),
            'valid_mask': tf.convert_to_tensor(np.expand_dims(valid_mask, axis=0), dtype=tf.float32)
        }



    def _rollback_current_request(self):
        """ Hàm giải cứu: Hoàn trả lại 100% băng thông đã tạm khấu trừ nếu Request thất bại """
        for u, v in self.touched_edges:
            self.current_bw_matrix[u, v] += self.bw_req
            self.current_bw_matrix[v, u] += self.bw_req
        # Xóa sạch vết sau khi đã khôi phục
        self.touched_edges.clear()

    def _calculate_jain_index(self):
        """ Tính chỉ số công bằng Jain trên toàn bộ các cạnh đang có cáp """
        # Tính độ hao mòn: Utilization = (Max - Current) / Max
        utilization = (self.max_capacity_matrix - self.current_bw_matrix) / (self.max_capacity_matrix + 1e-9)
        # Chỉ lấy giá trị ở những nơi thực sự có cáp nối (mask == 1)
        active_links = utilization[self.base_topology_mask == 1.0]
        if len(active_links) == 0:
            return 1.0
        sum_util = np.sum(active_links)
        sum_sq_util = np.sum(active_links**2) + 1e-9
        jain_index = (sum_util**2) / (len(active_links) * sum_sq_util)
        return jain_index

    def step(self, action):
        valid_mask = self._get_state_dict()['valid_mask'].numpy()[0]
        # Trường hợp 1: Chọn nhầm vào Node không hợp lệ ngay từ đầu
        if valid_mask[action] == 0.0:
            self._rollback_current_request() # Trả lại tài nguyên (nếu có)
            return self._get_state_dict(), -3.0, True, {'status': 'DeadEnd'}
        # --- TIẾN HÀNH TẠM ỨNG TÀI NGUYÊN ---
        self.current_bw_matrix[self.curr_node, action] -= self.bw_req
        self.current_bw_matrix[action, self.curr_node] -= self.bw_req
        # Ghi vết lại để chuẩn bị cho tình huống xấu nhất
        self.touched_edges.append((self.curr_node, action))
        edge_delay = self.static_delay_matrix[self.curr_node, action]
        self.remain_time -= edge_delay
        self.curr_node = action
        self.visited.append(action)
        next_state_dict = self._get_state_dict()
        next_valid_mask = next_state_dict['valid_mask'].numpy()[0]       
        # Trường hợp 2: Tới đích thành công mỹ mãn
        if self.curr_node == self.dst:
            # Giữ nguyên tài nguyên bị chiếm đóng trên mạng, xóa bộ nhớ tạm ứng
            self.touched_edges.clear()
            reward = 3.0 + self._calculate_jain_index() # Thưởng thêm nếu mạng đang khá cân bằng
            return next_state_dict, reward, True, {'status': 'Success'}
        # Trường hợp 3: Đâm vào ngõ cụt (Không còn đường nào hợp lệ để đi tiếp)
        elif np.sum(next_valid_mask) == 0.0:
            # THỰC HIỆN ROLLBACK TOÀN BỘ ĐƯỜNG ĐÃ ĐI CỦA REQUEST NÀY!
            reward = -3.0 + self._calculate_jain_index() # Phạt nặng nhưng vẫn có thể thưởng nếu mạng đang rất cân bằng
            self._rollback_current_request()
            return next_state_dict, reward, True, {'status': 'DeadEnd'}
        # Trường hợp 4: Vẫn đang luồn lách ổn định, chưa tới đích
        else:
            reward = 0.0
            return next_state_dict, reward, False, {'status': 'In_Progress'}