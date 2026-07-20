import matplotlib
matplotlib.use('Agg')
import tensorflow as tf

class BatchedPPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        
        self.c1 = 0.5  # Hệ số Loss Critic
        self.c2 = 0.01 # Hệ số Loss Entropy

    @tf.function
    def compute_gae_gpu(self, rewards, values, dones, next_v):
        """ 
        Tính GAE song song cho toàn bộ Batch.
        Đầu vào: [Time, Batch]
        """
        T = tf.shape(rewards)[0]
        B = tf.shape(rewards)[1]
        
        adv_array = tf.TensorArray(dtype=tf.float32, size=T)
        last_gae = tf.zeros([B], dtype=tf.float32)
        
        # Chèn next_v vào cuối mảng values để có thể gọi values[t+1]
        values_extended = tf.concat([values, tf.expand_dims(next_v, axis=0)], axis=0)
        
        for t in tf.range(T - 1, -1, -1):
            next_non_terminal = 1.0 - tf.cast(dones[t], tf.float32)
            
            delta = rewards[t] + self.gamma * values_extended[t+1] * next_non_terminal - values_extended[t]
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            
            adv_array = adv_array.write(t, last_gae)
            
        advs = adv_array.stack()
        returns = advs + values
        return advs, returns

    @tf.function
    def run_rollout_episode(self, env, num_steps=128):
        """ 
        Gom Data thuần túy trên GPU bằng TensorArray.
        1024 môi trường cùng chạy num_steps.
        """
        B = env.B
        
        # BỎ TỪ ĐIỂN, DÙNG BIẾN RỜI ĐỂ AUTOGRAPH THEO DÕI ĐƯỢC SCOPE
        ta_A_bw = tf.TensorArray(dtype=tf.float32, size=num_steps)
        ta_A_delay = tf.TensorArray(dtype=tf.float32, size=num_steps)
        ta_bw_req = tf.TensorArray(dtype=tf.float32, size=num_steps)
        ta_remain_time = tf.TensorArray(dtype=tf.float32, size=num_steps)
        ta_valid_mask = tf.TensorArray(dtype=tf.float32, size=num_steps)
        ta_curr_node = tf.TensorArray(dtype=tf.int32, size=num_steps)
        ta_dst_node = tf.TensorArray(dtype=tf.int32, size=num_steps)
            
        ta_actions = tf.TensorArray(dtype=tf.int32, size=num_steps)
        ta_logprobs = tf.TensorArray(dtype=tf.float32, size=num_steps)
        ta_rewards = tf.TensorArray(dtype=tf.float32, size=num_steps)
        ta_values = tf.TensorArray(dtype=tf.float32, size=num_steps)
        ta_dones = tf.TensorArray(dtype=tf.bool, size=num_steps)

        for t in tf.range(num_steps):
            state = env._get_state_dict()
            
            # Ghi state vào từng biến TensorArray rời rạc
            ta_A_bw = ta_A_bw.write(t, state['A_bw'])
            ta_A_delay = ta_A_delay.write(t, state['A_delay'])
            ta_bw_req = ta_bw_req.write(t, state['bw_req'])
            ta_remain_time = ta_remain_time.write(t, state['remain_time'])
            ta_valid_mask = ta_valid_mask.write(t, state['valid_mask'])
            ta_curr_node = ta_curr_node.write(t, state['curr_node'])
            ta_dst_node = ta_dst_node.write(t, state['dst_node'])
                
            # Model phán đoán
            action_probs, masked_logits, v_vals = self.model(state)
            actions = self.model.sample_action(masked_logits) # [B]
            
            # Tính logprob cho toàn bộ batch
            indices = tf.stack([tf.range(B), tf.cast(actions, tf.int32)], axis=1)
            action_prob_val = tf.gather_nd(action_probs, indices)
            logprobs = tf.math.log(action_prob_val + 1e-9)
            
            # Step môi trường
            next_state, rewards, dones = env.step(actions)
            
            # Ghi chép vết các thông số khác
            ta_actions = ta_actions.write(t, tf.cast(actions, tf.int32))
            ta_logprobs = ta_logprobs.write(t, logprobs)
            ta_rewards = ta_rewards.write(t, tf.cast(rewards, tf.float32))
            ta_values = ta_values.write(t, tf.squeeze(v_vals, axis=-1))
            ta_dones = ta_dones.write(t, dones)
            
        # --- BƯỚC BOOTSTRAPPING (Tầm nhìn tương lai cho những env chưa done) ---
        final_state = env._get_state_dict()
        _, _, next_v = self.model(final_state)
        next_v = tf.squeeze(next_v, axis=-1)
        
        # Đóng gói lại thành Dictionary SAU KHI vòng lặp đã kết thúc hoàn toàn
        stacked_states = {
            'A_bw': ta_A_bw.stack(),
            'A_delay': ta_A_delay.stack(),
            'bw_req': ta_bw_req.stack(),
            'remain_time': ta_remain_time.stack(),
            'valid_mask': ta_valid_mask.stack(),
            'curr_node': ta_curr_node.stack(),
            'dst_node': ta_dst_node.stack()
        }
        
        return (
            stacked_states,
            ta_actions.stack(),
            ta_logprobs.stack(),
            ta_rewards.stack(),
            ta_values.stack(),
            ta_dones.stack(),
            next_v
        )
    def make_dataset_generator(self, flat_states, flat_actions, flat_logprobs, flat_advs, flat_returns, batch_size):
        """ 
        Generator đẻ Minibatch siêu tốc chuẩn TensorFlow (Thay thế Dataloader/Buffer).
        Mọi thao tác shuffle và slice đều chạy trên GPU.
        """
        total_elements = tf.shape(flat_actions)[0]
        num_valid = (total_elements // batch_size) * batch_size
        
        # Trộn index
        perm = tf.random.shuffle(tf.range(num_valid, dtype=tf.int32))
        perm = tf.reshape(perm, [-1, batch_size])
        
        # Yield từng cụm data
        for i in range(tf.shape(perm)[0]):
            indices = perm[i]
            
            mb_states = {k: tf.gather(v, indices) for k, v in flat_states.items()}
            mb_actions = tf.gather(flat_actions, indices)
            mb_logprobs = tf.gather(flat_logprobs, indices)
            mb_advs = tf.gather(flat_advs, indices)
            mb_returns = tf.gather(flat_returns, indices)
            
            yield mb_states, mb_actions, mb_logprobs, mb_advs, mb_returns

    @tf.function
    def train_step(self, mb_states, mb_actions, mb_old_logprobs, mb_advs, mb_returns):
        """ Ép trọng số Model trên 1 Minibatch """
        
        # Chuẩn hóa GAE chống bùng nổ gradient
        mb_advs = (mb_advs - tf.reduce_mean(mb_advs)) / (tf.math.reduce_std(mb_advs) + 1e-8)
        
        with tf.GradientTape() as tape:
            # 1. Forward Pass
            action_probs, masked_logits, v_values = self.model(mb_states)
            v_values = tf.squeeze(v_values, axis=-1)
            
            # 2. Tìm lại xác suất của actions cũ
            batch_size = tf.shape(mb_actions)[0]
            indices = tf.stack([tf.range(batch_size), mb_actions], axis=1)
            new_probs = tf.gather_nd(action_probs, indices)
            new_logprobs = tf.math.log(new_probs + 1e-9)
            
            # 3. Tỷ lệ (Ratio)
            ratio = tf.exp(new_logprobs - mb_old_logprobs)
            
            # 4. Actor Loss (Clipped Surrogate)
            surr1 = ratio * mb_advs
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advs
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # 5. Critic Loss (MSE)
            critic_loss = tf.reduce_mean(tf.square(mb_returns - v_values))
            
            # 6. Entropy Loss
            entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-9), axis=-1))
            
            # 7. Total Loss
            total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
            
        # 8. Backpropagation & Cập nhật
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return actor_loss, critic_loss, entropy

    def learn(self, env, num_steps=128, ppo_epochs=4, minibatch_size=2048):
        """ 
        Hàm bao bọc (Wrapper) chạy luồng huấn luyện cho 1 Epoch tổng.
        Được gọi trực tiếp từ main.py
        """
        # 1. Rollout trên GPU
        (b_states, b_actions, b_logprobs, b_rewards, 
         b_values, b_dones, next_v) = self.run_rollout_episode(env, num_steps)
        
        # 2. Tính GAE
        adv_tensor, ret_tensor = self.compute_gae_gpu(b_rewards, b_values, b_dones, next_v)
        
        # 3. Flatten toàn bộ Tensor (Gộp chiều Time và chiều Batch)
        # SỬA LẠI DÒNG NÀY: Thêm .as_list() vào v.shape
        flat_states = {k: tf.reshape(v, [-1] + v.shape.as_list()[2:]) for k, v in b_states.items()}
        flat_actions = tf.reshape(b_actions, [-1])
        flat_logprobs = tf.reshape(b_logprobs, [-1])
        flat_advs = tf.reshape(adv_tensor, [-1])
        flat_returns = tf.reshape(ret_tensor, [-1])
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        batches_count = 0
        
        # 4. Train PPO n vòng
        for _ in range(ppo_epochs):
            dataset = self.make_dataset_generator(
                flat_states, flat_actions, flat_logprobs, flat_advs, flat_returns, minibatch_size
            )
            for mb_states, mb_actions, mb_logprobs, mb_advs, mb_returns in dataset:
                a_loss, c_loss, ent = self.train_step(mb_states, mb_actions, mb_logprobs, mb_advs, mb_returns)
                
                total_actor_loss += a_loss.numpy()
                total_critic_loss += c_loss.numpy()
                total_entropy += ent.numpy()
                batches_count += 1
                
        return {
            'actor_loss': total_actor_loss / batches_count,
            'critic_loss': total_critic_loss / batches_count,
            'entropy': total_entropy / batches_count
        }