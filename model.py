
import tensorflow as tf

class FiLMLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim=64):
        super(FiLMLayer, self).__init__()
        self.fc_gamma = tf.keras.layers.Dense(embed_dim, activation='sigmoid')
        self.fc_beta = tf.keras.layers.Dense(embed_dim)

    def call(self, H, condition):
        gamma = self.fc_gamma(condition) 
        beta = self.fc_beta(condition)   
        gamma = tf.expand_dims(gamma, axis=1) 
        beta = tf.expand_dims(beta, axis=1)   
        return (H * gamma) + beta

class GCNResidualLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim=64):
        super(GCNResidualLayer, self).__init__()
        self.W = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, A_norm, H):
        msg = tf.matmul(A_norm, H)
        out = self.W(msg)
        out = tf.nn.swish(out) # Dùng Swish cho mượt đạo hàm
        return self.layer_norm(out + H)

class RoutingPACModel(tf.keras.Model):
    def __init__(self,num_nodes=100, embed_dim=64, num_gcn_layers=4):
        super(RoutingPACModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.num_gcn_layers = num_gcn_layers
        
        self.node_embeddings_bw = tf.Variable(
            tf.random.normal([num_nodes, embed_dim]), trainable=True, name="node_emb_bw"
        )
        self.node_embeddings_delay = tf.Variable(
            tf.random.normal([num_nodes, embed_dim]), trainable=True, name="node_emb_delay"
        )
        
        self.bw_film = FiLMLayer(embed_dim)
        self.bw_gcns = [GCNResidualLayer(embed_dim) for _ in range(num_gcn_layers)]
        
        self.delay_film = FiLMLayer(embed_dim)
        self.delay_gcns = [GCNResidualLayer(embed_dim) for _ in range(num_gcn_layers)]
        
        self.actor_mixer = tf.keras.layers.Dense(embed_dim, activation='tanh')
        self.actor_query_proj = tf.keras.layers.Dense(embed_dim)
        self.actor_key_proj = tf.keras.layers.Dense(embed_dim)
        
        self.critic_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='tanh'),
            tf.keras.layers.Dense(128, activation='tanh'),
            tf.keras.layers.Dense(1) 
        ])

    def call(self, inputs):
        A_bw = inputs['A_bw']                 
        A_delay = inputs['A_delay']           
        batch_size = tf.shape(A_bw)[0]
        
        X_bw = tf.tile(tf.expand_dims(self.node_embeddings_bw, axis=0), [batch_size, 1, 1])
        X_delay = tf.tile(tf.expand_dims(self.node_embeddings_delay, axis=0), [batch_size, 1, 1])
        
        # BẢO VỆ DIMENSION: Ép cứng về [Batch, 1]
        bw_req = tf.reshape(inputs['bw_req'], [batch_size, 1])
        remain_time = tf.reshape(inputs['remain_time'], [batch_size, 1])
        
        curr_node = inputs['curr_node']       
        dst_node = inputs['dst_node']         
        valid_mask = inputs['valid_mask']     

        # --- DUAL-STREAM GCN ---
        H_bw = X_bw
        H_delay = X_delay
        
        for i in range(self.num_gcn_layers):
            H_bw = self.bw_film(H_bw, bw_req)
            H_bw = self.bw_gcns[i](A_bw, H_bw)
            
            H_delay = self.delay_film(H_delay, remain_time)
            H_delay = self.delay_gcns[i](A_delay, H_delay)

        # --- ACTOR ---
        A_set = tf.concat([H_bw, H_delay], axis=-1)
        A_mixed = self.actor_mixer(A_set)
        
        dst_indices = tf.expand_dims(dst_node, axis=1) 
        query = tf.gather_nd(A_mixed, dst_indices, batch_dims=1) 
        
        Q = self.actor_query_proj(query)          
        K = self.actor_key_proj(A_mixed)          
        
        Q = tf.expand_dims(Q, axis=2)             
        attention_logits = tf.squeeze(tf.matmul(K, Q), axis=-1) 
        scale_factor = tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        attention_logits = attention_logits / scale_factor
        
        valid_mask_f32 = tf.cast(valid_mask, tf.float32)
        masked_logits = attention_logits + (1.0 - valid_mask_f32) * -1e9
        action_probs = tf.nn.softmax(masked_logits, axis=-1)

        # --- CRITIC ---
        curr_indices = tf.expand_dims(curr_node, axis=1)
        
        global_bw = tf.reduce_mean(H_bw, axis=1)                 
        curr_bw = tf.gather_nd(H_bw, curr_indices, batch_dims=1) 
        dst_bw = tf.gather_nd(H_bw, dst_indices, batch_dims=1)   
        
        global_delay = tf.reduce_mean(H_delay, axis=1)                 
        curr_delay = tf.gather_nd(H_delay, curr_indices, batch_dims=1) 
        dst_delay = tf.gather_nd(H_delay, dst_indices, batch_dims=1)   
        
        critic_input = tf.concat([
            global_bw, curr_bw, dst_bw,
            global_delay, curr_delay, dst_delay
        ], axis=-1)
        
        v_value = self.critic_mlp(critic_input)

        return action_probs, masked_logits, v_value

    def sample_action(self, masked_logits):
        """
        Dùng thẳng masked_logits đã được lọc ở hàm call.
        Vừa nhanh vừa không bao giờ lo lỗi chia cho 0!
        """
        action = tf.random.categorical(masked_logits, num_samples=1)
        return tf.squeeze(action, axis=-1)