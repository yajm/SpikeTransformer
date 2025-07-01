import numpy as np
from typing import Dict, List, Tuple
from training_data import TRAINING_DATA
from constants import XAVIER_MULTIPLIER, XAVIER_MUTLITPLIER_2, EMBEDDING_INIT_SCALE, EMBEDDING_INIT_BIAS, QKV_INIT_SCALE, OUTPUT_INIT_SCALE, FF_INIT_SCALE, GRADIENT_CLIP_VALUE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, SURROGATE_CLIP_VALUE, SURROGATE_BETA, SPIKE_THRESH_EMBEDDING, SPIKE_THRESH_Q, SPIKE_THRESH_K, SPIKE_THRESH_V, SPIKE_THRESH_ATTN, SPIKE_THRESH_FF1, SPIKE_THRESH_FF2, FORWARD_MEMBRANE_CLIP_THRE, LEARNING_RATE, D_MODEL, N_HEADS, N_LAYERS, D_FF, TIMESTEPS, SHUFFLE_DATA, EPOCHS, LOG_INTERVAL, DECAY_FACTOR, RESET_VALUE

class SpikingNeuronLayer:
    """Leaky Integrate-and-Fire (LIF) spiking neuron layer"""
    
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.decay_factor = DECAY_FACTOR
        self.reset_value = RESET_VALUE
        self.membrane_potential = None
   
    def forward(self, x: np.ndarray, timestep: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        if self.membrane_potential is None or timestep == 0:
            self.membrane_potential = np.zeros_like(x)
        
        self.membrane_potential = self.membrane_potential * self.decay_factor + x
        self.membrane_potential = np.clip(self.membrane_potential, -FORWARD_MEMBRANE_CLIP_THRE, FORWARD_MEMBRANE_CLIP_THRE)
        spikes = (self.membrane_potential >= self.threshold).astype(np.float32)
        self.membrane_potential = np.where(
            spikes > 0, 
            self.reset_value, 
            self.membrane_potential
        )
        return spikes, self.membrane_potential.copy()


class SpikeTransformer:
    def __init__(self, 
                 max_seq_len: int = 20,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_ff: int = 256,
                 learning_rate: float = LEARNING_RATE,
                 timesteps: int = 4):
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.d_k = d_model // n_heads
        self.learning_rate = learning_rate
        self.timesteps = timesteps
        
        self.char_to_idx = {}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = 1
        for i, ascii_val in enumerate(range(32, 127)):
            self.char_to_idx[chr(ascii_val)] = i + 2
        self.vocab_size = len(self.char_to_idx)
        
        self.colors = ["blue", "red", "green", "yellow", "white", "black", "orange", "purple"]
        self.color_to_idx = {color: idx for idx, color in enumerate(self.colors)}

        self._initialize_parameters()
        self._initialize_spiking_layers()

        self.training_data = TRAINING_DATA
        self.spike_rates = {}
   
    def _initialize_parameters(self):
        """Initialize all model parameters"""
        # Embedding layer
        self.embedding = self._xavier_init(self.vocab_size, self.d_model) * EMBEDDING_INIT_SCALE
        self.embedding += EMBEDDING_INIT_BIAS
        self.pos_encoding = self._create_positional_encoding()
        
        # Initialize layers
        self.layers = []
        for _ in range(self.n_layers):
            layer = {
                # Attention weights
                'W_q': self._xavier_init(self.d_model, self.d_model) * QKV_INIT_SCALE,
                'W_k': self._xavier_init(self.d_model, self.d_model) * QKV_INIT_SCALE,
                'W_v': self._xavier_init(self.d_model, self.d_model) * QKV_INIT_SCALE,
                'W_o': self._xavier_init(self.d_model, self.d_model) * OUTPUT_INIT_SCALE,
                
                # Feed-forward
                'W_ff1': self._xavier_init(self.d_model, self.d_ff) * FF_INIT_SCALE,
                'b_ff1': np.zeros((1, self.d_ff)),
                'W_ff2': self._xavier_init(self.d_ff, self.d_model) * FF_INIT_SCALE,
                'b_ff2': np.zeros((1, self.d_model)),
                
                # Layer normalization parameters
                'ln1_gamma': np.ones((1, self.d_model)),
                'ln1_beta': np.zeros((1, self.d_model)),
                'ln2_gamma': np.ones((1, self.d_model)),
                'ln2_beta': np.zeros((1, self.d_model))
            }
            self.layers.append(layer)
        
        # Output projection
        self.W_out = self._xavier_init(self.d_model, len(self.colors)) * OUTPUT_INIT_SCALE
        self.b_out = np.zeros((1, len(self.colors)))
        
        # Initialize optimizer states
        self._init_optimizer_states()

    def _initialize_spiking_layers(self):
        self.spike_layers = {}
        
        self.spike_layers['embedding'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_EMBEDDING)
        
        for i in range(self.n_layers):
            self.spike_layers[f'layer_{i}_q'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_Q)
            self.spike_layers[f'layer_{i}_k'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_K)
            self.spike_layers[f'layer_{i}_v'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_V)
            self.spike_layers[f'layer_{i}_attn'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_ATTN)
            self.spike_layers[f'layer_{i}_ff1'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_FF1)
            self.spike_layers[f'layer_{i}_ff2'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_FF2)
    
    def _xavier_init(self, n_in, n_out):
        return np.random.randn(n_in, n_out) * np.sqrt(XAVIER_MULTIPLIER / n_in) * XAVIER_MUTLITPLIER_2
    
    def _create_positional_encoding(self):
        """Create sinusoidal positional encoding"""
        pe = np.zeros((self.max_seq_len, self.d_model))
        position = np.arange(0, self.max_seq_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(10 / self.d_model))      
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    
    def _init_optimizer_states(self):
        """Initialize Adam optimizer states"""
        self.adam_m = {}
        self.adam_v = {}
        self.adam_t = 0
        
        # Initialize states for all parameters
        params = ['embedding', 'W_out', 'b_out']
        for param in params:
            self.adam_m[param] = np.zeros_like(getattr(self, param))
            self.adam_v[param] = np.zeros_like(getattr(self, param))
        
        # Layer parameters
        for i, layer in enumerate(self.layers):
            for param_name, param_value in layer.items():
                key = f'layer_{i}_{param_name}'
                self.adam_m[key] = np.zeros_like(param_value)
                self.adam_v[key] = np.zeros_like(param_value)
    
    def spike_driven_self_attention(self, x: np.ndarray, layer: Dict, layer_idx: int, 
                               mask: np.ndarray = None, timestep: int = 0) -> Tuple[np.ndarray, Dict]:
        Q_membrane = np.matmul(x, layer['W_q'])
        K_membrane = np.matmul(x, layer['W_k'])
        V_membrane = np.matmul(x, layer['W_v'])
        
        # Convert to spikes
        Q_spikes, _ = self.spike_layers[f'layer_{layer_idx}_q'].forward(Q_membrane, timestep)
        K_spikes, _ = self.spike_layers[f'layer_{layer_idx}_k'].forward(K_membrane, timestep)
        V_spikes, _ = self.spike_layers[f'layer_{layer_idx}_v'].forward(V_membrane, timestep)
        
        # Record spike rates
        self.spike_rates[f'layer_{layer_idx}_Q'] = np.mean(Q_spikes)
        self.spike_rates[f'layer_{layer_idx}_K'] = np.mean(K_spikes)
        self.spike_rates[f'layer_{layer_idx}_V'] = np.mean(V_spikes)
        
        # Original SDSA-1: Q ⊗ (K ⊗ V)
        KV_hadamard = K_spikes * V_spikes
        KV_sum = np.sum(KV_hadamard, axis=2, keepdims=True)
        attention_spikes, _ = self.spike_layers[f'layer_{layer_idx}_attn'].forward(KV_sum, timestep)
        output = V_spikes * attention_spikes 
        
        if mask is not None:
            output = output * mask[:, :, None]
        return np.matmul(output, layer['W_o'])
    
    def spike_driven_feed_forward(self, x: np.ndarray, layer: Dict, layer_idx: int, 
                                 timestep: int = 0) -> np.ndarray:
        hidden_membrane = np.matmul(x, layer['W_ff1']) + layer['b_ff1']
        hidden_spikes, _ = self.spike_layers[f'layer_{layer_idx}_ff1'].forward(hidden_membrane, timestep)
        output_membrane = np.matmul(hidden_spikes, layer['W_ff2']) + layer['b_ff2']
        _, output_membrane = self.spike_layers[f'layer_{layer_idx}_ff2'].forward(
            output_membrane, timestep
        )
        return output_membrane
    
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                   eps: float = 1e-6) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def spike_transformer_block(self, x_membrane: np.ndarray, layer: Dict, layer_idx: int,
                               mask: np.ndarray = None, timestep: int = 0) -> Tuple[np.ndarray, Dict]:
        residual = x_membrane
        attn_output = self.spike_driven_self_attention(x_membrane, layer, layer_idx, mask, timestep)
        x_membrane = residual + attn_output
        x_membrane = self.layer_norm(x_membrane, layer['ln1_gamma'], layer['ln1_beta'])
        residual = x_membrane
        ff_output = self.spike_driven_feed_forward(x_membrane, layer, layer_idx, timestep)
        x_membrane = residual + ff_output
        x_membrane = self.layer_norm(x_membrane, layer['ln2_gamma'], layer['ln2_beta'])
        
        return x_membrane


    def encode_word(self, word: str) -> np.ndarray:
        """Convert word to indices with padding"""
        word = word.lower()[:self.max_seq_len]
        indices = []
        
        for char in word:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx['<UNK>'])
        
        # Pad to max length
        while len(indices) < self.max_seq_len:
            indices.append(self.char_to_idx['<PAD>'])
        
        return np.array(indices)
    
    def create_padding_mask(self, x: np.ndarray) -> np.ndarray:
        """Create mask for padded positions"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return (x != self.char_to_idx['<PAD>']).astype(np.float32)
    
    def forward(self, word: str) -> Tuple[np.ndarray, List]:
        self.spike_rates = {}
        x = self.encode_word(word)
        mask = self.create_padding_mask(x)
        x = x.reshape(1, -1)
        mask = mask.reshape(1, -1)
        output_accumulator = None
  
        for t in range(self.timesteps):
            x_embed = self.embedding[x]
            seq_len = x_embed.shape[1]
            x_embed = x_embed + self.pos_encoding[:seq_len, :]
            _, x_membrane = self.spike_layers['embedding'].forward(x_embed, t)
            x_current = x_membrane
            
            for i, layer in enumerate(self.layers):
                x_current = self.spike_transformer_block(x_current, layer, i, mask, t)

            mask_expanded = mask[:, :, None]
            x_pooled = np.sum(x_current * mask_expanded, axis=1) / np.sum(mask_expanded, axis=1)
            if output_accumulator is None:
                output_accumulator = x_pooled
            else:
                output_accumulator += x_pooled
        
        output_avg = output_accumulator / self.timesteps
        logits = np.matmul(output_avg, self.W_out) + self.b_out
        probs = self.softmax(logits.flatten())
        
        return probs
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def predict(self, word: str) -> Tuple[str, np.ndarray]:
        probs = self.forward(word)
        pred_idx = np.argmax(probs)
        return self.colors[pred_idx], probs
    
    def compute_loss(self, word: str, target_color: str) -> float:
        probs = self.forward(word)
        target_idx = self.color_to_idx[target_color]
        loss = -np.log(probs[target_idx] + 1e-10)
        return loss

    def adam_update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        grad = np.clip(grad, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE)
        self.adam_t += 1
        beta1, beta2 = ADAM_BETA1, ADAM_BETA2
        eps = ADAM_EPSILON
        
        self.adam_m[param_name] = beta1 * self.adam_m[param_name] + (1 - beta1) * grad
        self.adam_v[param_name] = beta2 * self.adam_v[param_name] + (1 - beta2) * grad**2
        
        m_hat = self.adam_m[param_name] / (1 - beta1**self.adam_t)
        v_hat = self.adam_v[param_name] / (1 - beta2**self.adam_t)
        
        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
    
    def backward(self, word: str, target_color: str) -> float:
        self.adam_t += 1
        all_activations = []
        
        # ============ FORWARD PASS WITH ACTIVATION TRACKING ============
        x = self.encode_word(word)
        mask = self.create_padding_mask(x)
        x = x.reshape(1, -1)
        mask = mask.reshape(1, -1)
        x_indices = x[0]
        
        for t in range(self.timesteps):
            activations = {}
            activations['timestep'] = t
            
            # Embedding lookup
            x_embed = self.embedding[x_indices].reshape(1, -1, self.d_model)
            activations['embed_input'] = x_embed.copy()
            
            # Add positional encoding
            seq_len = x_embed.shape[1]
            x_embed = x_embed + self.pos_encoding[:seq_len, :]
            activations['embed_with_pos'] = x_embed.copy()
            
            # Convert embeddings to spikes
            embed_spikes, embed_membrane = self.spike_layers['embedding'].forward(x_embed, t)
            activations['embed_spikes'] = embed_spikes.copy()
            activations['embed_membrane'] = embed_membrane.copy()
            
            # Use membrane potential as input (membrane shortcuts)
            layer_membrane = embed_membrane
            
            # Track through each transformer layer
            for i, layer in enumerate(self.layers):
                # Store input membrane potential
                activations[f'layer_{i}_input_membrane'] = layer_membrane.copy()
                
                # === Spike-Driven Self-Attention ===
                # Generate Q, K, V membrane potentials
                Q_membrane = np.matmul(layer_membrane, layer['W_q'])
                K_membrane = np.matmul(layer_membrane, layer['W_k'])
                V_membrane = np.matmul(layer_membrane, layer['W_v'])
                activations[f'layer_{i}_Q_membrane'] = Q_membrane.copy()
                activations[f'layer_{i}_K_membrane'] = K_membrane.copy()
                activations[f'layer_{i}_V_membrane'] = V_membrane.copy()
                
                # Convert to spikes
                Q_spikes, _ = self.spike_layers[f'layer_{i}_q'].forward(Q_membrane, t)
                K_spikes, _ = self.spike_layers[f'layer_{i}_k'].forward(K_membrane, t)
                V_spikes, _ = self.spike_layers[f'layer_{i}_v'].forward(V_membrane, t)
                activations[f'layer_{i}_Q_spikes'] = Q_spikes.copy()
                activations[f'layer_{i}_K_spikes'] = K_spikes.copy()
                activations[f'layer_{i}_V_spikes'] = V_spikes.copy()
                
    
                KV_hadamard = K_spikes * V_spikes
                activations[f'layer_{i}_KV_hadamard'] = KV_hadamard.copy()
                
                KV_sum = np.sum(KV_hadamard, axis=2, keepdims=True)
                activations[f'layer_{i}_KV_sum'] = KV_sum.copy()
                
                attn_spikes, _ = self.spike_layers[f'layer_{i}_attn'].forward(KV_sum, t)
                activations[f'layer_{i}_attn_spikes'] = attn_spikes.copy()
                
                # Use Q_spikes (fixed from V_spikes)
                attn_output_spikes = V_spikes * attn_spikes
                activations[f'layer_{i}_attn_output_spikes'] = attn_output_spikes.copy()
                

                # Apply mask if provided
                if mask is not None:
                    attn_output_spikes = attn_output_spikes * mask[:, :, None]
                
                # Output projection
                attn_output = np.matmul(attn_output_spikes, layer['W_o'])
                activations[f'layer_{i}_attn_output'] = attn_output.copy()
                
                attn_residual = layer_membrane + attn_output
                activations[f'layer_{i}_attn_residual'] = attn_residual.copy()
                
                # LayerNorm 1
                ln1_output = self.layer_norm(attn_residual, layer['ln1_gamma'], layer['ln1_beta'])
                activations[f'layer_{i}_ln1_output'] = ln1_output.copy()
                
                # === Spike-Driven Feed-forward ===
                # FF layer 1
                ff_hidden_membrane = np.matmul(ln1_output, layer['W_ff1']) + layer['b_ff1']
                activations[f'layer_{i}_ff_hidden_membrane'] = ff_hidden_membrane.copy()
                
                # Convert to spikes
                ff_hidden_spikes, _ = self.spike_layers[f'layer_{i}_ff1'].forward(
                    ff_hidden_membrane, t
                )
                activations[f'layer_{i}_ff_hidden_spikes'] = ff_hidden_spikes.copy()
                
                # FF layer 2
                ff_output_membrane = np.matmul(ff_hidden_spikes, layer['W_ff2']) + layer['b_ff2']
                activations[f'layer_{i}_ff_output_membrane'] = ff_output_membrane.copy()
                
                # Convert to spikes (but we use membrane for residual)
                _, ff_output_mem_after = self.spike_layers[f'layer_{i}_ff2'].forward(
                    ff_output_membrane, t
                )
                activations[f'layer_{i}_ff_output_mem_after'] = ff_output_mem_after.copy()
                
                # Membrane shortcut for FF
                ff_residual = ln1_output + ff_output_mem_after
                activations[f'layer_{i}_ff_residual'] = ff_residual.copy()
                
                # LayerNorm 2
                layer_output = self.layer_norm(ff_residual, layer['ln2_gamma'], layer['ln2_beta'])
                activations[f'layer_{i}_output'] = layer_output.copy()
                
                layer_membrane = layer_output
            
            # Global average pooling
            mask_expanded = mask[:, :, None]
            x_pooled = np.sum(layer_membrane * mask_expanded, axis=1) / np.sum(mask_expanded, axis=1)
            activations['pooled'] = x_pooled.copy()
            
            # Store activations for this timestep
            all_activations.append(activations)
            
            # Accumulate pooled output
            if t == 0:
                accumulated_output = x_pooled
            else:
                accumulated_output += x_pooled
        
        # Average over timesteps
        final_output = accumulated_output / self.timesteps
        
        # Output projection
        logits = np.matmul(final_output, self.W_out) + self.b_out
        probs = self.softmax(logits.flatten())
        
        # Compute loss
        target_idx = self.color_to_idx[target_color]
        loss = -np.log(probs[target_idx] + 1e-10)
        
        # ============ BACKWARD PASS ============
        grads = {}

        for i in range(self.n_layers):
            for param in ['W_q', 'W_k', 'W_v', 'W_o', 'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2',
                        'ln1_gamma', 'ln1_beta', 'ln2_gamma', 'ln2_beta']:
                grads[f'layer_{i}_{param}'] = 0

        for char_idx in range(self.vocab_size):
            grads[f'embedding_{char_idx}'] = np.zeros(self.d_model)

        grad_logits = probs.copy()
        grad_logits[target_idx] -= 1.0
        grad_logits = grad_logits.reshape(1, -1)
        
        # 2. Output layer gradients
        grads['W_out'] = np.matmul(final_output.T, grad_logits)
        grads['b_out'] = grad_logits
        
        # Update output layer immediately
        self.W_out = self.adam_update('W_out', self.W_out, grads['W_out'])
        self.b_out = self.adam_update('b_out', self.b_out, grads['b_out'])
        
        # 3. Gradient w.r.t. averaged pooled features
        grad_pooled_avg = np.matmul(grad_logits, self.W_out.T)
        
        # Process each timestep in reverse
        for t in reversed(range(self.timesteps)):
            activations = all_activations[t]
            
            # Gradient for this timestep's pooled output
            grad_pooled = grad_pooled_avg / self.timesteps
            
            # 4. Gradient through pooling
            grad_seq = np.zeros_like(activations[f'layer_{self.n_layers-1}_output'])
            seq_len = grad_seq.shape[1]
            n_active = np.sum(mask[0])
            for i in range(seq_len):
                if mask[0, i] > 0:
                    grad_seq[0, i] = grad_pooled[0] / n_active
            
            # 5. Backward through transformer layers
            grad_layer_membrane = grad_seq
            
            for i in reversed(range(self.n_layers)):
                layer = self.layers[i]
                
                # === Backward through LayerNorm 2 ===
                ln2_input = activations[f'layer_{i}_ff_residual']
                grad_ln2_input, grad_ln2_gamma, grad_ln2_beta = self.layer_norm_backward(
                    grad_layer_membrane, ln2_input, layer['ln2_gamma']
                )
                
                grads[f'layer_{i}_ln2_gamma'] += grad_ln2_gamma
                grads[f'layer_{i}_ln2_beta'] += grad_ln2_beta
                
                # === Backward through FF residual ===
                grad_ff_mem_after = grad_ln2_input
                grad_ln1_output = grad_ln2_input
                
                # === Backward through spiking FF layer 2 ===
                ff_output_membrane = activations[f'layer_{i}_ff_output_membrane']
                spike_grad_ff2 = self.surrogate_gradient(ff_output_membrane, SPIKE_THRESH_FF2)
                grad_ff_output_membrane = grad_ff_mem_after * spike_grad_ff2
                
                # Gradient through FF layer 2 linear transform
                ff_hidden_spikes = activations[f'layer_{i}_ff_hidden_spikes']
                ff_hidden_spikes_2d = ff_hidden_spikes.reshape(-1, self.d_ff)
                grad_ff_output_2d = grad_ff_output_membrane.reshape(-1, self.d_model)
                
                grad_W_ff2 = np.matmul(ff_hidden_spikes_2d.T, grad_ff_output_2d)
                grad_b_ff2 = np.sum(grad_ff_output_membrane, axis=(0, 1), keepdims=True).reshape(1, -1)
                
                grads[f'layer_{i}_W_ff2'] += grad_W_ff2
                grads[f'layer_{i}_b_ff2'] += grad_b_ff2
                
                grad_ff_hidden_spikes = np.matmul(grad_ff_output_membrane, layer['W_ff2'].T)
                
                # === Backward through spiking FF layer 1 ===
                ff_hidden_membrane = activations[f'layer_{i}_ff_hidden_membrane']
                spike_grad_ff1 = self.surrogate_gradient(ff_hidden_membrane, SPIKE_THRESH_FF1)
                grad_ff_hidden_membrane = grad_ff_hidden_spikes * spike_grad_ff1
                
                # Gradient through FF layer 1 linear transform
                ln1_output = activations[f'layer_{i}_ln1_output']
                ln1_output_2d = ln1_output.reshape(-1, self.d_model)
                grad_ff_hidden_2d = grad_ff_hidden_membrane.reshape(-1, self.d_ff)
                
                grad_W_ff1 = np.matmul(ln1_output_2d.T, grad_ff_hidden_2d)
                grad_b_ff1 = np.sum(grad_ff_hidden_membrane, axis=(0, 1), keepdims=False).reshape(1, -1)
                
                grads[f'layer_{i}_W_ff1'] += grad_W_ff1
                grads[f'layer_{i}_b_ff1'] += grad_b_ff1
                
                grad_ln1_output += np.matmul(grad_ff_hidden_membrane, layer['W_ff1'].T)
                
                # === Backward through LayerNorm 1 ===
                ln1_input = activations[f'layer_{i}_attn_residual']
                grad_ln1_input, grad_ln1_gamma, grad_ln1_beta = self.layer_norm_backward(
                    grad_ln1_output, ln1_input, layer['ln1_gamma']
                )
                
                grads[f'layer_{i}_ln1_gamma'] += grad_ln1_gamma
                grads[f'layer_{i}_ln1_beta'] += grad_ln1_beta
                
                # === Backward through attention residual ===
                grad_attn_output = grad_ln1_input
                grad_layer_input_membrane = grad_ln1_input
                
                # === Backward through attention output projection ===
                attn_output_spikes = activations[f'layer_{i}_attn_output_spikes']
                grad_W_o = np.matmul(
                    attn_output_spikes.reshape(-1, self.d_model).T,
                    grad_attn_output.reshape(-1, self.d_model)
                )
                grads[f'layer_{i}_W_o'] += grad_W_o
                
                grad_attn_output_spikes = np.matmul(grad_attn_output, layer['W_o'].T)
                
                Q_spikes = activations[f'layer_{i}_Q_spikes']
                K_spikes = activations[f'layer_{i}_K_spikes']
                V_spikes = activations[f'layer_{i}_V_spikes']
                attn_spikes = activations[f'layer_{i}_attn_spikes']
                
                # Gradient flows through Q (not V)
                grad_Q_spikes = grad_attn_output_spikes * attn_spikes
                
                # Gradient w.r.t attn_spikes
                grad_attn_spikes = np.sum(grad_attn_output_spikes * Q_spikes, axis=2, keepdims=True) 
  
                KV_sum = activations[f'layer_{i}_KV_sum']
                spike_grad_attn = self.surrogate_gradient(KV_sum, SPIKE_THRESH_ATTN)
                grad_KV_sum = grad_attn_spikes * spike_grad_attn
                
                grad_KV_hadamard = np.zeros_like(activations[f'layer_{i}_KV_hadamard'])
                grad_KV_hadamard = np.broadcast_to(grad_KV_sum, grad_KV_hadamard.shape)
                
                grad_K_spikes = grad_KV_hadamard * V_spikes
                grad_V_spikes = grad_KV_hadamard * K_spikes
                
                # Convert to membrane gradients
                Q_membrane = activations[f'layer_{i}_Q_membrane']
                K_membrane = activations[f'layer_{i}_K_membrane']
                V_membrane = activations[f'layer_{i}_V_membrane']
                
                spike_grad_Q = self.surrogate_gradient(Q_membrane, SPIKE_THRESH_Q)
                spike_grad_K = self.surrogate_gradient(K_membrane, SPIKE_THRESH_K)
                spike_grad_V = self.surrogate_gradient(V_membrane, SPIKE_THRESH_V)
                
                grad_Q_membrane = grad_Q_spikes * spike_grad_Q
                grad_K_membrane = grad_K_spikes * spike_grad_K
                grad_V_membrane = grad_V_spikes * spike_grad_V
                
                # Compute weight gradients
                layer_input_membrane = activations[f'layer_{i}_input_membrane']
                layer_input_2d = layer_input_membrane.reshape(-1, self.d_model)
                
                grad_W_q = np.matmul(layer_input_2d.T, grad_Q_membrane.reshape(-1, self.d_model))
                grad_W_k = np.matmul(layer_input_2d.T, grad_K_membrane.reshape(-1, self.d_model))
                grad_W_v = np.matmul(layer_input_2d.T, grad_V_membrane.reshape(-1, self.d_model))
                
                grads[f'layer_{i}_W_q'] += grad_W_q
                grads[f'layer_{i}_W_k'] += grad_W_k
                grads[f'layer_{i}_W_v'] += grad_W_v
                
                grad_from_attention = (np.matmul(grad_Q_membrane, layer['W_q'].T) +
                                    np.matmul(grad_K_membrane, layer['W_k'].T) +
                                    np.matmul(grad_V_membrane, layer['W_v'].T))
                
                grad_layer_membrane = grad_layer_input_membrane + grad_from_attention
            
            embed_membrane = activations['embed_membrane']
            spike_grad_embed = self.surrogate_gradient(
                activations['embed_with_pos'], SPIKE_THRESH_EMBEDDING
            )
            grad_embed_with_pos = grad_layer_membrane * spike_grad_embed
            
            grad_embed = grad_embed_with_pos
            
            for i, char_idx in enumerate(x_indices):
                if char_idx != self.char_to_idx['<PAD>']:
                    embedding_grad = grad_embed[0, i]
                    grads[f'embedding_{char_idx}'] += embedding_grad
        
        for i, layer in enumerate(self.layers):
            for param_name in ['W_q', 'W_k', 'W_v', 'W_o', 'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2',
                            'ln1_gamma', 'ln1_beta', 'ln2_gamma', 'ln2_beta']:
                param_key = f'layer_{i}_{param_name}'
                if param_key in grads:
                    avg_grad = grads[param_key] / self.timesteps
                    layer[param_name] = self.adam_update(param_key, layer[param_name], avg_grad)
        
        for char_idx in range(self.vocab_size):
            param_name = f'embedding_{char_idx}'
            if param_name in grads:
                # Initialize Adam states if needed
                if param_name not in self.adam_m:
                    self.adam_m[param_name] = np.zeros(self.d_model)
                    self.adam_v[param_name] = np.zeros(self.d_model)
                
                avg_grad = grads[param_name] / self.timesteps
                
                beta1, beta2 = ADAM_BETA1, ADAM_BETA2
                eps = ADAM_EPSILON
                self.adam_m[param_name] = beta1 * self.adam_m[param_name] + (1 - beta1) * avg_grad
                self.adam_v[param_name] = beta2 * self.adam_v[param_name] + (1 - beta2) * avg_grad**2
                m_hat = self.adam_m[param_name] / (1 - beta1**self.adam_t)
                v_hat = self.adam_v[param_name] / (1 - beta2**self.adam_t)
                self.embedding[char_idx] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        return loss


    def surrogate_gradient(self, x: np.ndarray, threshold: float, beta: float = SURROGATE_BETA) -> np.ndarray:
        centered = beta * (x - threshold)
        centered = np.clip(centered, -SURROGATE_CLIP_VALUE, SURROGATE_CLIP_VALUE)
        sigmoid = 1 / (1 + np.exp(-centered))
        return beta * sigmoid * (1 - sigmoid)

    def layer_norm_backward(self, grad_output, x, gamma, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        x_norm = (x - mean) / std
        
        N = x.shape[-1]
        grad_x_norm = grad_output * gamma
        grad_var = np.sum(grad_x_norm * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=-1, keepdims=True)
        grad_mean = np.sum(grad_x_norm * -1 / std, axis=-1, keepdims=True) + grad_var * np.sum(-2 * (x - mean), axis=-1, keepdims=True) / N
        grad_x = grad_x_norm / std + grad_var * 2 * (x - mean) / N + grad_mean / N
        
        axes_to_sum = tuple(range(grad_output.ndim - 1))
        grad_gamma = np.sum(grad_output * x_norm, axis=axes_to_sum, keepdims=False)
        grad_beta = np.sum(grad_output, axis=axes_to_sum, keepdims=False)
        
        grad_gamma = grad_gamma.reshape(1, -1)
        grad_beta = grad_beta.reshape(1, -1)
        
        return grad_x, grad_gamma, grad_beta

    def train(self, epochs: int = 100):
        """Train the spike-driven transformer"""
        print(f"Training Spike-Driven Transformer with {len(self.training_data)} examples")
        print(f"Architecture: {self.n_layers} layers, {self.n_heads} heads, {self.d_model} dimensions, Timesteps: {self.timesteps}")
        print("="*60)
        
        training_pairs = list(self.training_data.items())

        initial_lr = self.learning_rate
    
        for epoch in range(epochs):
            self.learning_rate = initial_lr * (1 + np.cos(np.pi * epoch / epochs)) / 2
            if SHUFFLE_DATA:
                np.random.shuffle(training_pairs)
            total_loss = 0
            correct = 0
            avg_spike_rate = 0
            
            for word, target_color in training_pairs:
                loss = self.backward(word, target_color)
                total_loss += loss
                
                pred_color, _ = self.predict(word)
                if pred_color == target_color:
                    correct += 1

                if self.spike_rates:
                    avg_spike_rate += np.mean(list(self.spike_rates.values()))
            
            avg_spike_rate /= len(training_pairs)
            
            if epoch % LOG_INTERVAL == 0:
                accuracy = correct / len(training_pairs) * 100
                avg_loss = total_loss / len(training_pairs)
                print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.1f}%, "
                      f"Avg Spike Rate = {avg_spike_rate:.3f}")
        
        print("\nTraining complete!")

    def analyze_efficiency(self):
        """Analyze the efficiency of spike-driven computation"""
        print("\n" + "="*60)
        print("SPIKE-DRIVEN EFFICIENCY ANALYSIS")
        print("="*60)
        
        # Test on a sample word
        test_word = "water"
        _ = self.forward(test_word)
        
        print(f"\nSpike rates for '{test_word}':")
        for layer_name, rate in sorted(self.spike_rates.items()):
            print(f"  {layer_name}: {rate:.3f}")
        
        # Calculate theoretical energy savings
        avg_spike_rate = np.mean(list(self.spike_rates.values()))
        vanilla_ops = self.n_layers * (3 * self.max_seq_len * self.d_model**2 +  # Q,K,V projections
                                      2 * self.max_seq_len**2 * self.d_model)    # Attention
        spike_ops = vanilla_ops * avg_spike_rate  # Only active neurons compute
        
        print(f"\nTheoretical energy savings:")
        print(f"  Average spike rate: {avg_spike_rate:.3f}")
        print(f"  Vanilla transformer ops: {vanilla_ops:,}")
        print(f"  Spike-driven ops: {spike_ops:,.0f}")
        print(f"  Energy reduction: {(1 - spike_ops/vanilla_ops)*100:.1f}%")

    def test_generalization(self):
        """Test on unseen words"""
        print("\n" + "="*60)
        print("Testing on UNSEEN words")
        print("="*60)
        
        test_words = {
            "waterfall": "blue", "firetruck": "red", "grassland": "green",
            "sunshine": "yellow", "snowflake": "white", "darkness": "black",
        }
        
        correct = 0
        total_spike_rate = 0
        
        for word, expected in test_words.items():
            pred_color, probs = self.predict(word)
            confidence = np.max(probs) * 100
            _ = self.forward(word)
            spike_rate = np.mean(list(self.spike_rates.values()))
            total_spike_rate += spike_rate
            
            status = "✓" if pred_color == expected else "✗"
            if pred_color == expected:
                correct += 1
            
            print(f"{word:12} -> {pred_color:8} ({confidence:4.1f}%, spike_rate={spike_rate:.3f}) "
                  f"[Expected: {expected:8}] {status}")
        
        accuracy = correct / len(test_words) * 100
        avg_spike_rate = total_spike_rate / len(test_words)
        print(f"\nGeneralization accuracy: {correct}/{len(test_words)} ({accuracy:.1f}%)")
        print(f"Average spike rate: {avg_spike_rate:.3f}")


if __name__ == "__main__":
    print("Spike-driven Transformer from scratch using only Numpy\n")
    
    model = SpikeTransformer(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        timesteps=TIMESTEPS,
        learning_rate=LEARNING_RATE
    )
    
    print("\n" + "="*60)
    print("Before training (random initialization):")
    test_words = ["water", "fire", "grass", "night", "sunshine"]
    for word in test_words:
        color, probs = model.predict(word)
        print(f"  {word:10} -> {color:8} ({np.max(probs)*100:.1f}%)")

    print("\n" + "="*60)
    model.train(epochs=EPOCHS)
    
    print("\n" + "="*60)
    print("Performance on TRAINING words:")
    print("="*60)
    for word in ["ocean", "cherry", "forest", "banana", "snow", "midnight"]:
        pred, probs = model.predict(word)
        conf = np.max(probs) * 100
        true = model.training_data.get(word, "?")
        print(f"{word:10} -> {pred:8} ({conf:4.1f}%) [True: {true}]")
    
    model.test_generalization()
    model.analyze_efficiency()

    while True:
        word = input("\nEnter a word (or 'quit'): ").strip()
        if word.lower() == 'quit':
            break
        
        pred_color, probs = model.predict(word)
        _ = model.forward(word)
        avg_spike_rate = np.mean(list(model.spike_rates.values()))
        
        print(f"\n'{word}' -> {pred_color}")
        print(f"Average spike rate: {avg_spike_rate:.3f} (lower = more efficient)")
        print(f"\nConfidence scores:")
        
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices[:3]:
            color = model.colors[idx]
            prob = probs[idx]
            bar = "█" * int(prob * 30)
            print(f"  {color:8}: {bar:30} {prob*100:5.1f}%")