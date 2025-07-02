import json
import numpy as np
from typing import Dict, List, Tuple
from constants import XAVIER_MULTIPLIER, XAVIER_MUTLITPLIER_2, EMBEDDING_INIT_SCALE, EMBEDDING_INIT_BIAS, QKV_INIT_SCALE, OUTPUT_INIT_SCALE, FF_INIT_SCALE, GRADIENT_CLIP_VALUE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, SURROGATE_CLIP_VALUE, SURROGATE_BETA, SPIKE_THRESH_EMBEDDING, SPIKE_THRESH_Q, SPIKE_THRESH_K, SPIKE_THRESH_V, SPIKE_THRESH_ATTN, SPIKE_THRESH_FF1, SPIKE_THRESH_FF2, FORWARD_MEMBRANE_CLIP_THRE, LEARNING_RATE, D_MODEL, N_HEADS, N_LAYERS, D_FF, TIMESTEPS, EPOCHS, LOG_INTERVAL, DECAY_FACTOR, RESET_VALUE, MAX_SEQ_LEN, SAVE_FILEPATH, TRAIN_MODEL
import random
import os
from tqdm import tqdm
import pickle
import gc

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

class WikipediaDataLoader:
    """Handles loading and preprocessing of Wikipedia articles"""
    
    def __init__(self, json_path: str, max_article_length: int = 10000, 
                 chunk_size: int = 512, overlap: int = 64):
        self.json_path = json_path
        self.max_article_length = max_article_length
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.articles = []
        self.chunks = []
        
    def load_articles(self):
        """Load articles from JSON file"""
        print(f"Loading Wikipedia articles from {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract content from each article
        for article in tqdm(data, desc="Processing articles"):
            if 'content' in article and article['content']:
                # Take only first max_article_length characters to manage memory
                content = article['content'][:self.max_article_length]
                self.articles.append({
                    'title': article.get('title', 'Unknown'),
                    'content': content
                })
        
        print(f"Loaded {len(self.articles)} articles")
        
    def create_chunks(self):
        """Split articles into overlapping chunks for training"""
        print("Creating training chunks...")
        
        for article in tqdm(self.articles, desc="Chunking articles"):
            content = article['content']
            
            # Skip very short articles
            if len(content) < 100:
                continue
                
            # Create overlapping chunks
            for i in range(0, len(content) - self.chunk_size + 1, self.chunk_size - self.overlap):
                chunk = content[i:i + self.chunk_size]
                if len(chunk) >= 100:  # Minimum chunk size
                    self.chunks.append(chunk)
        
        print(f"Created {len(self.chunks)} training chunks")
        
    def get_training_chunks(self, num_chunks: int = None):
        """Get training chunks, optionally limiting the number"""
        if num_chunks:
            return random.sample(self.chunks, min(num_chunks, len(self.chunks)))
        return self.chunks

class SpikeLLM():  
    def __init__(self, ): 
        self.d_model = D_MODEL
        self.n_heads = N_HEADS
        self.n_layers = N_LAYERS
        self.d_ff = D_FF
        self.max_seq_len = MAX_SEQ_LEN
        self.timesteps = TIMESTEPS
        self.learning_rate = LEARNING_RATE
        
        # Initialize transformer layers
        self.layers = []
        for i in range(self.n_layers):
            layer = {
                'W_q': self._xavier_init(self.d_model, self.d_model) * QKV_INIT_SCALE,
                'W_k': self._xavier_init(self.d_model, self.d_model) * QKV_INIT_SCALE,
                'W_v': self._xavier_init(self.d_model, self.d_model) * QKV_INIT_SCALE,
                'W_o': self._xavier_init(self.d_model, self.d_model) * OUTPUT_INIT_SCALE,
                'W_ff1': self._xavier_init(self.d_model, self.d_ff) * FF_INIT_SCALE,
                'b_ff1': np.zeros((1, self.d_ff)),
                'W_ff2': self._xavier_init(self.d_ff, self.d_model) * FF_INIT_SCALE,
                'b_ff2': np.zeros((1, self.d_model)),
                'ln1_gamma': np.ones((1, self.d_model)),
                'ln1_beta': np.zeros((1, self.d_model)),
                'ln2_gamma': np.ones((1, self.d_model)),
                'ln2_beta': np.zeros((1, self.d_model))
            }
            self.layers.append(layer)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding()
        
        # Initialize Adam optimizer states
        self.adam_t = 0
        
        # Initialize spiking layers
        self._initialize_spiking_layers()
        
        # NOW do the vocab setup (rest of your existing __init__ code)
        self.training_history = {
            'epoch': [],
            'loss': [],
            'perplexity': []
        }

        self.vocab = [
            '<PAD>',    # 0
            '<START>',  # 1 
            '<END>',    # 2
            '<UNK>',    # 3
            '<CAP>',    # 4 - Capitalization marker
            ' ',        # 5 - Space
            '\n',       # 6 - Newline
            # Lowercase letters by frequency (26 chars)
            'e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r', 'd', 'l', 'c', 'u', 'm', 
            'f', 'p', 'g', 'w', 'y', 'b', 'v', 'k', 'x', 'z', 'j', 'q',
            # Numbers
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            # Essential punctuation
            '.', ',', "'", '-', ':', '(', ')', ';', '!', '"',
            # Special chars
            '@', '#', '$', '%', '&', '*', '+', '=', '_'
        ]
        
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        # Re-initialize weights for new vocab size
        self.embedding = self._xavier_init(self.vocab_size, self.d_model) * EMBEDDING_INIT_SCALE
        self.embedding += EMBEDDING_INIT_BIAS
        self.W_out = self._xavier_init(self.d_model, self.vocab_size) * OUTPUT_INIT_SCALE
        self.b_out = np.zeros((1, self.vocab_size))
        
        # Re-initialize optimizer states for new parameters
        self._init_optimizer_states()

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

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _create_positional_encoding(self):
        """Create sinusoidal positional encoding"""
        pe = np.zeros((self.max_seq_len, self.d_model))
        position = np.arange(0, self.max_seq_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(10 / self.d_model))      
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def adam_update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        grad = np.clip(grad, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE)
        beta1, beta2 = ADAM_BETA1, ADAM_BETA2
        eps = ADAM_EPSILON
        
        self.adam_m[param_name] = beta1 * self.adam_m[param_name] + (1 - beta1) * grad
        self.adam_v[param_name] = beta2 * self.adam_v[param_name] + (1 - beta2) * grad**2
        
        m_hat = self.adam_m[param_name] / (1 - beta1**self.adam_t)
        v_hat = self.adam_v[param_name] / (1 - beta2**self.adam_t)
        
        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                   eps: float = 1e-6) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
        
    def create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal attention mask to prevent attending to future positions"""
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask
    
    def forward(self, input_ids: np.ndarray, target_ids: np.ndarray = None, return_activations: bool = False):
        """
        Forward pass for LLM (no pooling, returns logits for each position)
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            return_activations: Whether to return activation dict for backward pass
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            activations: List of activation dicts per timestep (if requested)
        """
        batch_size, seq_len = input_ids.shape
        self.spike_rates = {}
        all_activations = [] if return_activations else None
        

        # Accumulator for multi-timestep outputs
        output_accumulator = None
        
        for t in range(self.timesteps):
            activations = {'timestep': t} if return_activations else None
            
            # Embedding lookup
            x_embed = self.embedding[input_ids]  # [batch_size, seq_len, d_model]
            if return_activations:
                activations['embed_input'] = x_embed.copy()
            
            # Add positional encoding
            x_embed = x_embed + self.pos_encoding[:seq_len, :]
            if return_activations:
                activations['embed_with_pos'] = x_embed.copy()
            
            # Convert to spikes
            embed_spikes, embed_membrane = self.spike_layers['embedding'].forward(x_embed, t)
            if return_activations:
                activations['embed_spikes'] = embed_spikes.copy()
                activations['embed_membrane'] = embed_membrane.copy()
            
            # Process through transformer layers
            layer_membrane = embed_membrane
            
            for i, layer in enumerate(self.layers):
                if return_activations:
                    activations[f'layer_{i}_input_membrane'] = layer_membrane.copy()
                
                # Self-attention with causal mask
                layer_membrane = self.spike_transformer_block(
                    layer_membrane, layer, i, t, activations if return_activations else None
                )
   
                if return_activations:
                    activations[f'layer_{i}_output'] = layer_membrane.copy()
            
            # Output projection (no pooling for LLM)
            if output_accumulator is None:
                output_accumulator = layer_membrane
            else:
                output_accumulator += layer_membrane
            
            if return_activations:
                all_activations.append(activations)
        
        # Average over timesteps
        final_output = output_accumulator / self.timesteps
        
        # Project to vocabulary size
        logits = np.matmul(final_output, self.W_out) + self.b_out
        
        if target_ids is None:
            return logits
        
        # Compute loss
        batch_size, seq_len = input_ids.shape
        total_loss = 0.0
        n_valid_tokens = 0
        
        # Compute softmax probabilities
        probs = np.zeros_like(logits)
        for i in range(seq_len):
            probs[:, i, :] = self.softmax(logits[:, i, :], axis=-1)
        
        # Cross-entropy loss
        for b in range(batch_size):
            for i in range(seq_len):
                if target_ids[b, i] != self.char_to_idx['<PAD>']:
                    total_loss -= np.log(probs[b, i, target_ids[b, i]] + 1e-10)
                    n_valid_tokens += 1
        
        avg_loss = total_loss / max(n_valid_tokens, 1)
        
        return logits, avg_loss, all_activations
    
    def spike_transformer_block(self, x_membrane: np.ndarray, layer: Dict, layer_idx: int,
                                 timestep: int, activations: Dict = None):
        """Modified transformer block with causal masking and activation tracking"""
        residual = x_membrane
        
        # Self-attention
        attn_output = self.spike_driven_self_attention(
            x_membrane, layer, layer_idx, timestep, activations
        )
        x_membrane = residual + attn_output
        x_membrane = self.layer_norm(x_membrane, layer['ln1_gamma'], layer['ln1_beta'])
        
        if activations is not None:
            activations[f'layer_{layer_idx}_ln1_output'] = x_membrane.copy()
        
        residual = x_membrane
        
        # Feed-forward
        ff_output = self.spike_driven_feed_forward(
            x_membrane, layer, layer_idx, timestep, activations
        )
        x_membrane = residual + ff_output
        x_membrane = self.layer_norm(x_membrane, layer['ln2_gamma'], layer['ln2_beta'])
        
        return x_membrane
    
    def spike_driven_self_attention(self, x: np.ndarray, layer: Dict, layer_idx: int,
                                        timestep: int, activations: Dict = None):
        # Generate Q, K, V
        Q_membrane = np.matmul(x, layer['W_q'])
        K_membrane = np.matmul(x, layer['W_k'])
        V_membrane = np.matmul(x, layer['W_v'])
        
        if activations is not None:
            activations[f'layer_{layer_idx}_Q_membrane'] = Q_membrane.copy()
            activations[f'layer_{layer_idx}_K_membrane'] = K_membrane.copy()
            activations[f'layer_{layer_idx}_V_membrane'] = V_membrane.copy()
        
        # Convert to spikes
        Q_spikes, _ = self.spike_layers[f'layer_{layer_idx}_q'].forward(Q_membrane, timestep)
        K_spikes, _ = self.spike_layers[f'layer_{layer_idx}_k'].forward(K_membrane, timestep)
        V_spikes, _ = self.spike_layers[f'layer_{layer_idx}_v'].forward(V_membrane, timestep)
        
        if activations is not None:
            activations[f'layer_{layer_idx}_Q_spikes'] = Q_spikes.copy()
            activations[f'layer_{layer_idx}_K_spikes'] = K_spikes.copy()
            activations[f'layer_{layer_idx}_V_spikes'] = V_spikes.copy()
        
        # Spike-driven attention with causal mask
        KV_hadamard = K_spikes * V_spikes  # [batch, seq_len, d_model]
        
        # Apply causal mask before summing
        KV_hadamard_masked = KV_hadamard
        
        
        if activations is not None:
            activations[f'layer_{layer_idx}_KV_hadamard'] = KV_hadamard.copy()
            activations[f'layer_{layer_idx}_KV_hadamard_masked'] = KV_hadamard_masked.copy()
        
        KV_sum = np.sum(KV_hadamard_masked, axis=1, keepdims=True)  # [batch, 1, d_model]
        
        if activations is not None:
            activations[f'layer_{layer_idx}_KV_sum'] = KV_sum.copy()
        
        attn_spikes, _ = self.spike_layers[f'layer_{layer_idx}_attn'].forward(KV_sum, timestep)
        
        if activations is not None:
            activations[f'layer_{layer_idx}_attn_spikes'] = attn_spikes.copy()
        
        # Apply attention to values
        attn_output_spikes = V_spikes * attn_spikes
        
        if activations is not None:
            activations[f'layer_{layer_idx}_attn_output_spikes'] = attn_output_spikes.copy()
        
        # Output projection
        return np.matmul(attn_output_spikes, layer['W_o'])
    
    def spike_driven_feed_forward(self, x: np.ndarray, layer: Dict, layer_idx: int,
                                     timestep: int, activations: Dict = None):
        """Feed-forward with activation tracking"""
        # First layer
        hidden_membrane = np.matmul(x, layer['W_ff1']) + layer['b_ff1']
        if activations is not None:
            activations[f'layer_{layer_idx}_ff_hidden_membrane'] = hidden_membrane.copy()
        
        hidden_spikes, _ = self.spike_layers[f'layer_{layer_idx}_ff1'].forward(hidden_membrane, timestep)
        if activations is not None:
            activations[f'layer_{layer_idx}_ff_hidden_spikes'] = hidden_spikes.copy()
        
        # Second layer
        output_membrane = np.matmul(hidden_spikes, layer['W_ff2']) + layer['b_ff2']
        if activations is not None:
            activations[f'layer_{layer_idx}_ff_output_membrane'] = output_membrane.copy()
        
        _, output_mem_after = self.spike_layers[f'layer_{layer_idx}_ff2'].forward(output_membrane, timestep)
        if activations is not None:
            activations[f'layer_{layer_idx}_ff_output_mem_after'] = output_mem_after.copy()
        
        return output_mem_after
    
    def backward(self, input_ids: np.ndarray, logits: np.ndarray, target_ids: np.ndarray, avg_loss: float, all_activations: List[Dict]):
        """
        Backward pass using pre-computed forward results
        
        Args:
            logits: [batch_size, seq_len, vocab_size] from forward
            target_ids: [batch_size, seq_len] target tokens
            avg_loss: scalar loss from forward
            all_activations: list of activation dicts from forward
        """
        self.adam_t += 1
        batch_size, seq_len = target_ids.shape
        
        # Initialize gradients
        grads = {}
        
        # Initialize layer gradients
        for i in range(self.n_layers):
            for param in ['W_q', 'W_k', 'W_v', 'W_o', 'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2',
                         'ln1_gamma', 'ln1_beta', 'ln2_gamma', 'ln2_beta']:
                grads[f'layer_{i}_{param}'] = 0
        
        # Initialize embedding gradients
        for token_idx in range(self.vocab_size):
            grads[f'embedding_{token_idx}'] = np.zeros(self.d_model)
        
        # 1. Gradient w.r.t. logits (initial gradient)

        probs = np.zeros_like(logits)
        for i in range(seq_len):
            probs[:, i, :] = self.softmax(logits[:, i, :], axis=-1)
        
        grad_logits = probs.copy()
        for b in range(batch_size):
            for i in range(seq_len):
                if target_ids[b, i] != self.char_to_idx['<PAD>']:
                    grad_logits[b, i, target_ids[b, i]] -= 1.0
                else:
                    grad_logits[b, i, :] = 0.0
        
        # 2. Output layer gradients
        final_output = np.zeros((batch_size, seq_len, self.d_model))
        for t in range(self.timesteps):
            layer_output = all_activations[t][f'layer_{self.n_layers-1}_output']
            final_output += layer_output / self.timesteps
        
        # Accumulate gradients from all positions
        grad_W_out = np.zeros_like(self.W_out)
        grad_b_out = np.zeros_like(self.b_out)
        
        for i in range(seq_len):
            grad_W_out += np.matmul(final_output[:, i, :].T, grad_logits[:, i, :])
            grad_b_out += np.sum(grad_logits[:, i, :], axis=0, keepdims=True)
        
        grads['W_out'] = grad_W_out
        grads['b_out'] = grad_b_out
        
        # Update output layer
        self.W_out = self.adam_update('W_out', self.W_out, grad_W_out)
        self.b_out = self.adam_update('b_out', self.b_out, grad_b_out)
        
        # 3. Gradient w.r.t. final transformer output
        grad_final_output = np.matmul(grad_logits, self.W_out.T)
        
        # 4. Process each timestep in reverse
        for t in reversed(range(self.timesteps)):
            activations = all_activations[t]
            
            # Gradient for this timestep
            grad_layer_output = grad_final_output / self.timesteps
            
            # 5. Backward through transformer layers
            for i in reversed(range(self.n_layers)):
                layer = self.layers[i]
                
                # === Backward through LayerNorm 2 ===
                ln2_input = activations[f'layer_{i}_ff_residual'] if f'layer_{i}_ff_residual' in activations else \
                           activations[f'layer_{i}_ln1_output'] + activations[f'layer_{i}_ff_output_mem_after']
                
                grad_ln2_input, grad_ln2_gamma, grad_ln2_beta = self.layer_norm_backward(
                    grad_layer_output, ln2_input, layer['ln2_gamma']
                )
                
                grads[f'layer_{i}_ln2_gamma'] += grad_ln2_gamma
                grads[f'layer_{i}_ln2_beta'] += grad_ln2_beta
                
                # === Backward through FF residual connection ===
                grad_ff_mem_after = grad_ln2_input
                grad_ln1_output_from_ff = grad_ln2_input
                
                # === Backward through FF layer 2 ===
                ff_output_membrane = activations[f'layer_{i}_ff_output_membrane']
                spike_grad_ff2 = self.surrogate_gradient(ff_output_membrane, SPIKE_THRESH_FF2)
                grad_ff_output_membrane = grad_ff_mem_after * spike_grad_ff2
                
                # Gradient through FF2 linear transform
                ff_hidden_spikes = activations[f'layer_{i}_ff_hidden_spikes']
                
                # Reshape for matrix multiplication
                grad_ff2_input = grad_ff_output_membrane.reshape(-1, self.d_model)
                ff_hidden_flat = ff_hidden_spikes.reshape(-1, self.d_ff)
                
                grad_W_ff2 = np.matmul(ff_hidden_flat.T, grad_ff2_input)
                grad_b_ff2 = np.sum(grad_ff_output_membrane, axis=(0, 1), keepdims=True)
                
                grads[f'layer_{i}_W_ff2'] += grad_W_ff2
                grads[f'layer_{i}_b_ff2'] += grad_b_ff2
                
                grad_ff_hidden_spikes = np.matmul(grad_ff_output_membrane, layer['W_ff2'].T)
                
                # === Backward through FF layer 1 ===
                ff_hidden_membrane = activations[f'layer_{i}_ff_hidden_membrane']
                spike_grad_ff1 = self.surrogate_gradient(ff_hidden_membrane, SPIKE_THRESH_FF1)
                grad_ff_hidden_membrane = grad_ff_hidden_spikes * spike_grad_ff1
                
                # Gradient through FF1 linear transform
                ln1_output = activations[f'layer_{i}_ln1_output']
                
                grad_ff1_input = grad_ff_hidden_membrane.reshape(-1, self.d_ff)
                ln1_flat = ln1_output.reshape(-1, self.d_model)
                
                grad_W_ff1 = np.matmul(ln1_flat.T, grad_ff1_input)
                grad_b_ff1 = np.sum(grad_ff_hidden_membrane, axis=(0, 1), keepdims=True)
                
                grads[f'layer_{i}_W_ff1'] += grad_W_ff1
                grads[f'layer_{i}_b_ff1'] += grad_b_ff1
                
                grad_ln1_output = grad_ln1_output_from_ff + np.matmul(grad_ff_hidden_membrane, layer['W_ff1'].T)
                
                # === Backward through LayerNorm 1 ===
                attn_residual = activations[f'layer_{i}_input_membrane'] + \
                               np.matmul(activations[f'layer_{i}_attn_output_spikes'], layer['W_o'])
                
                grad_attn_residual, grad_ln1_gamma, grad_ln1_beta = self.layer_norm_backward(
                    grad_ln1_output, attn_residual, layer['ln1_gamma']
                )
                
                grads[f'layer_{i}_ln1_gamma'] += grad_ln1_gamma
                grads[f'layer_{i}_ln1_beta'] += grad_ln1_beta
                
                # === Backward through attention residual ===
                grad_attn_output = grad_attn_residual
                grad_layer_input = grad_attn_residual
                
                # === Backward through attention output projection ===
                attn_output_spikes = activations[f'layer_{i}_attn_output_spikes']
                
                grad_attn_flat = grad_attn_output.reshape(-1, self.d_model)
                attn_spikes_flat = attn_output_spikes.reshape(-1, self.d_model)
                
                grad_W_o = np.matmul(attn_spikes_flat.T, grad_attn_flat)
                grads[f'layer_{i}_W_o'] += grad_W_o
                
                grad_attn_output_spikes = np.matmul(grad_attn_output, layer['W_o'].T)
                
                # === Backward through spike-driven attention ===
                V_spikes = activations[f'layer_{i}_V_spikes']
                attn_spikes = activations[f'layer_{i}_attn_spikes']
                
                # Gradient w.r.t V_spikes
                grad_V_spikes = grad_attn_output_spikes * attn_spikes
                
                # Gradient w.r.t attn_spikes
                grad_attn_spikes = np.sum(grad_attn_output_spikes * V_spikes, axis=2, keepdims=True)
                
                # Backward through attention spike generation
                KV_sum = activations[f'layer_{i}_KV_sum']
                spike_grad_attn = self.surrogate_gradient(KV_sum, SPIKE_THRESH_ATTN)
                grad_KV_sum = grad_attn_spikes * spike_grad_attn
                
                grad_KV_hadamard = np.broadcast_to(grad_KV_sum, (batch_size, seq_len, self.d_model))
                
                # Gradients for K and V spikes
                K_spikes = activations[f'layer_{i}_K_spikes']
                grad_K_spikes = grad_KV_hadamard * V_spikes
                grad_V_spikes += grad_KV_hadamard * K_spikes
                
                # Convert spike gradients to membrane gradients
                Q_membrane = activations[f'layer_{i}_Q_membrane']
                K_membrane = activations[f'layer_{i}_K_membrane']
                V_membrane = activations[f'layer_{i}_V_membrane']
                
                spike_grad_Q = self.surrogate_gradient(Q_membrane, SPIKE_THRESH_Q)
                spike_grad_K = self.surrogate_gradient(K_membrane, SPIKE_THRESH_K)
                spike_grad_V = self.surrogate_gradient(V_membrane, SPIKE_THRESH_V)
                
                # Note: Q gradient comes from being unused in SDSA-1 architecture
                grad_Q_membrane = np.zeros_like(Q_membrane)  # Q is not used in output
                grad_K_membrane = grad_K_spikes * spike_grad_K
                grad_V_membrane = grad_V_spikes * spike_grad_V
                
                # Compute weight gradients for Q, K, V
                layer_input = activations[f'layer_{i}_input_membrane']
                layer_input_flat = layer_input.reshape(-1, self.d_model)
                
                grad_W_q = np.matmul(layer_input_flat.T, grad_Q_membrane.reshape(-1, self.d_model))
                grad_W_k = np.matmul(layer_input_flat.T, grad_K_membrane.reshape(-1, self.d_model))
                grad_W_v = np.matmul(layer_input_flat.T, grad_V_membrane.reshape(-1, self.d_model))
                
                grads[f'layer_{i}_W_q'] += grad_W_q
                grads[f'layer_{i}_W_k'] += grad_W_k
                grads[f'layer_{i}_W_v'] += grad_W_v
                
                # Gradient to previous layer
                grad_from_attention = (np.matmul(grad_Q_membrane, layer['W_q'].T) +
                                     np.matmul(grad_K_membrane, layer['W_k'].T) +
                                     np.matmul(grad_V_membrane, layer['W_v'].T))
                
                grad_layer_output = grad_layer_input + grad_from_attention
            
            # === Backward through embeddings ===
            embed_with_pos = activations['embed_with_pos']
            spike_grad_embed = self.surrogate_gradient(embed_with_pos, SPIKE_THRESH_EMBEDDING)
            grad_embed_with_pos = grad_layer_output * spike_grad_embed
            
            # Remove positional encoding gradient
            grad_embed = grad_embed_with_pos  # Positional encoding is fixed
            
            # Accumulate embedding gradients
            for b in range(batch_size):
                for i in range(seq_len):
                    token_idx = input_ids[b, i]
                    if token_idx != self.char_to_idx['<PAD>']:
                        grads[f'embedding_{token_idx}'] += grad_embed[b, i, :]
        
        # ============ UPDATE PARAMETERS ============
        # Update layer parameters
        for i, layer in enumerate(self.layers):
            for param_name in ['W_q', 'W_k', 'W_v', 'W_o', 'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2',
                              'ln1_gamma', 'ln1_beta', 'ln2_gamma', 'ln2_beta']:
                param_key = f'layer_{i}_{param_name}'
                if param_key in grads:
                    # Average gradient over timesteps
                    avg_grad = grads[param_key] / self.timesteps
                    layer[param_name] = self.adam_update(param_key, layer[param_name], avg_grad)
        
        # Update embeddings
        for token_idx in range(self.vocab_size):
            param_name = f'embedding_{token_idx}'
            if param_name in grads and np.any(grads[param_name] != 0):
                # Initialize Adam states if needed
                if param_name not in self.adam_m:
                    self.adam_m[param_name] = np.zeros(self.d_model)
                    self.adam_v[param_name] = np.zeros(self.d_model)
                
                # Average gradient over timesteps
                avg_grad = grads[param_name] / self.timesteps
                
                # Adam update
                beta1, beta2 = ADAM_BETA1, ADAM_BETA2
                eps = ADAM_EPSILON
                self.adam_m[param_name] = beta1 * self.adam_m[param_name] + (1 - beta1) * avg_grad
                self.adam_v[param_name] = beta2 * self.adam_v[param_name] + (1 - beta2) * avg_grad**2
                m_hat = self.adam_m[param_name] / (1 - beta1**self.adam_t)
                v_hat = self.adam_v[param_name] / (1 - beta2**self.adam_t)
                self.embedding[token_idx] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        return avg_loss
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> str:
        """Generate text given a prompt using the trained model"""
        # Tokenize prompt
        tokens = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in prompt]
        tokens = [self.char_to_idx['<START>']] + tokens
        
        generated = prompt
        
        for _ in range(max_length):
            # Prepare input (pad if necessary)
            input_tokens = tokens[-self.max_seq_len:]
            if len(input_tokens) < self.max_seq_len:
                input_tokens = input_tokens + [self.char_to_idx['<PAD>']] * (self.max_seq_len - len(input_tokens))
            
            input_ids = np.array([input_tokens])
            
            logits = self.forward(input_ids)
            
            # Get prediction for the last valid position
            last_pos = min(len(tokens) - 1, self.max_seq_len - 1)
            next_logits = logits[0, last_pos, :] / temperature
            
            # Sample next token
            probs = self.softmax(next_logits)
            next_token = np.random.choice(self.vocab_size, p=probs)
            
            # Stop if END token
            if next_token == self.char_to_idx['<END>']:
                break
            
            # Add to sequence
            tokens.append(next_token)
            if next_token != self.char_to_idx['<PAD>']:
                generated += self.idx_to_char[next_token]
        
        return generated

    def preprocess_text(self, text: str) -> List[int]:
        """Convert text to token indices with proper handling of unknown characters"""
        tokens = []
        for char in text:
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                # Use PAD token for unknown characters
                tokens.append(self.char_to_idx['<PAD>'])
        return tokens
    
    def create_training_batch(self, texts: List[str], batch_size: int = 8):
        """Create batches of training data"""
        batch_x = []
        batch_y = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                # Tokenize with START and END tokens
                tokens = [self.char_to_idx['<START>']]
                tokens += self.preprocess_text(text[:self.max_seq_len-2])
                tokens += [self.char_to_idx['<END>']]
                
                # Pad to max_seq_len
                while len(tokens) < self.max_seq_len:
                    tokens.append(self.char_to_idx['<PAD>'])
                
                # Truncate if necessary
                tokens = tokens[:self.max_seq_len]
                
                # Create input (all but last) and target (all but first)
                batch_x.append(tokens[:-1])
                batch_y.append(tokens[1:])
        
        return np.array(batch_x), np.array(batch_y)
    
    def compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from loss"""
        return np.exp(min(loss, 100))  # Cap to prevent overflow
    
    def train_on_wikipedia(self, data_loader: WikipediaDataLoader, 
                          epochs: int = 10, 
                          batch_size: int = 8,
                          chunks_per_epoch: int = 1000,
                          checkpoint_dir: str = './checkpoints',
                          save_interval: int = 5):
        """Train on Wikipedia dataset with proper batching and checkpointing"""
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"Starting training on Wikipedia dataset")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Chunks per epoch: {chunks_per_epoch}")
        
        for epoch in range(epochs):
            # Get random subset of chunks for this epoch
            epoch_chunks = data_loader.get_training_chunks(chunks_per_epoch)
            random.shuffle(epoch_chunks)
            
            epoch_loss = 0
            n_batches = 0
            
            # Create progress bar for this epoch
            pbar = tqdm(range(0, len(epoch_chunks), batch_size), 
                       desc=f"Epoch {epoch+1}/{epochs}")
            
            for i in pbar:
                # Get batch of texts
                batch_texts = epoch_chunks[i:i + batch_size]
                
                # Create training batch
                batch_x, batch_y = self.create_training_batch(batch_texts, batch_size)
                
                batch_loss = 0
                if len(batch_x) > 0:
                    logits, loss, activations = self.forward(batch_x, batch_y, return_activations=True)
                    self.backward(batch_x, logits, batch_y, loss, activations)
                    batch_loss = loss
                

                epoch_loss += batch_loss
                n_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'perplexity': f'{self.compute_perplexity(batch_loss):.2f}'
                })
                
                # Garbage collection every 100 batches
                if n_batches % 100 == 0:
                    gc.collect()
            
            # Epoch statistics
            avg_epoch_loss = epoch_loss / n_batches
            epoch_perplexity = self.compute_perplexity(avg_epoch_loss)
            
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['loss'].append(avg_epoch_loss)
            self.training_history['perplexity'].append(epoch_perplexity)
            
            print(f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Perplexity: {epoch_perplexity:.2f}")
            
            # Generate sample
            if (epoch + 1) % LOG_INTERVAL == 0:
                prompts = ["The ", "In ", "A ", "Wikipedia "]
                for prompt in prompts:
                    generated = self.generate(prompt, max_length=100, temperature=0.8)
                    print(f"Sample '{prompt}': {generated[:200]}...")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(checkpoint_dir, epoch + 1)
    
    def save_checkpoint(self, checkpoint_dir: str, epoch: int):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(checkpoint_dir, f'spike_llm_epoch_{epoch}.pkl')
        
        checkpoint = {
            'epoch': epoch,
            'model_params': {
                'embedding': self.embedding,
                'W_out': self.W_out,
                'b_out': self.b_out,
                'layers': self.layers,
                'pos_encoding': self.pos_encoding
            },
            'optimizer_states': {
                'adam_m': self.adam_m,
                'adam_v': self.adam_v,
                'adam_t': self.adam_t
            },
            'training_history': self.training_history,
            'config': {
                'max_seq_len': self.max_seq_len,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'd_ff': self.d_ff,
                'timesteps': self.timesteps,
                'learning_rate': self.learning_rate
            }
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore model parameters
        self.embedding = checkpoint['model_params']['embedding']
        self.W_out = checkpoint['model_params']['W_out']
        self.b_out = checkpoint['model_params']['b_out']
        self.layers = checkpoint['model_params']['layers']
        self.pos_encoding = checkpoint['model_params']['pos_encoding']
        
        # Restore optimizer states
        self.adam_m = checkpoint['optimizer_states']['adam_m']
        self.adam_v = checkpoint['optimizer_states']['adam_v']
        self.adam_t = checkpoint['optimizer_states']['adam_t']
        
        # Restore training history
        self.training_history = checkpoint['training_history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

if __name__ == "__main__":
    WIKIPEDIA_JSON_PATH = "wikipedia_top_articles.json"
    
    if TRAIN_MODEL:
        # Initialize data loader
        data_loader = WikipediaDataLoader(
            json_path=WIKIPEDIA_JSON_PATH,
            max_article_length=10000,
            chunk_size=512,
            overlap=64
        )
        
        # Load and process data
        data_loader.load_articles()
        data_loader.create_chunks()
        
        # Initialize model
        model = SpikeLLM()
        
        # Train model
        model.train_on_wikipedia(
            data_loader=data_loader,
            epochs=EPOCHS,
            batch_size=8,
            chunks_per_epoch=1000,
            checkpoint_dir=SAVE_FILEPATH,
            save_interval=5
        )
    else:
        # Load and test model
        model = SpikeLLM()
        checkpoint_path = os.path.join(SAVE_FILEPATH, 'spike_llm_epoch_10.pkl')
        if os.path.exists(checkpoint_path):
            model.load_checkpoint(checkpoint_path)
            
            # Generate samples
            prompts = ["The ", "In the ", "Wikipedia ", "A "]
            for prompt in prompts:
                generated = model.generate(prompt, max_length=200, temperature=0.8)
                print(f"\nPrompt: '{prompt}'")
                print(f"Generated: {generated}")