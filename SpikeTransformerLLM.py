import numpy as np
import pickle
from typing import Dict, List, Tuple
import random

# Constants (keeping most of your original ones)
XAVIER_MULTIPLIER = 2.0
XAVIER_MULTIPLIER_2 = 1.0
EMBEDDING_INIT_SCALE = 0.5
EMBEDDING_INIT_BIAS = 0.0
QKV_INIT_SCALE = 0.5
OUTPUT_INIT_SCALE = 0.1
FF_INIT_SCALE = 0.5
GRADIENT_CLIP_VALUE = 1.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
SURROGATE_CLIP_VALUE = 10.0
SURROGATE_BETA = 5.0
SPIKE_THRESH_EMBEDDING = 0.5
SPIKE_THRESH_Q = 0.5
SPIKE_THRESH_K = 0.5
SPIKE_THRESH_V = 0.5
SPIKE_THRESH_ATTN = 0.5
SPIKE_THRESH_FF1 = 0.5
SPIKE_THRESH_FF2 = 0.5
FORWARD_MEMBRANE_CLIP_THRE = 5.0
LEARNING_RATE = 0.001
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 4
D_FF = 512
TIMESTEPS = 4
DECAY_FACTOR = 0.9
RESET_VALUE = 0.0
MAX_SEQ_LEN = 256
BATCH_SIZE = 8
EPOCHS = 100
LOG_INTERVAL = 10

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


class SpikeLLM:
    def __init__(self, 
                 max_seq_len: int = 256,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 d_ff: int = 512,
                 learning_rate: float = 0.001,
                 timesteps: int = 4):
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.d_k = d_model // n_heads
        self.learning_rate = learning_rate
        self.timesteps = timesteps
        
        # Character vocabulary (keeping your approach)
        self.char_to_idx = {}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<START>'] = 1
        self.char_to_idx['<END>'] = 2
        for i, ascii_val in enumerate(range(32, 127)):
            self.char_to_idx[chr(ascii_val)] = i + 3
        self.vocab_size = len(self.char_to_idx)
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        self._initialize_parameters()
        self._initialize_spiking_layers()
        self.spike_rates = {}
   
    def _initialize_parameters(self):
        """Initialize all model parameters"""
        # Embedding layer
        self.embedding = self._xavier_init(self.vocab_size, self.d_model) * EMBEDDING_INIT_SCALE
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
        
        # Output projection for next token prediction
        self.W_out = self._xavier_init(self.d_model, self.vocab_size) * OUTPUT_INIT_SCALE
        self.b_out = np.zeros((1, self.vocab_size))
        
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
        return np.random.randn(n_in, n_out) * np.sqrt(XAVIER_MULTIPLIER / n_in) * XAVIER_MULTIPLIER_2
    
    def _create_positional_encoding(self):
        """Create sinusoidal positional encoding"""
        pe = np.zeros((self.max_seq_len, self.d_model))
        position = np.arange(0, self.max_seq_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
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
    
    def create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal mask to prevent looking at future tokens"""
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask
    
    def spike_driven_self_attention(self, x: np.ndarray, layer: Dict, layer_idx: int, 
                                   causal_mask: np.ndarray = None, timestep: int = 0) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        Q_membrane = np.matmul(x, layer['W_q'])
        K_membrane = np.matmul(x, layer['W_k'])
        V_membrane = np.matmul(x, layer['W_v'])
        
        # Convert to spikes
        Q_spikes, _ = self.spike_layers[f'layer_{layer_idx}_q'].forward(Q_membrane, timestep)
        K_spikes, _ = self.spike_layers[f'layer_{layer_idx}_k'].forward(K_membrane, timestep)
        V_spikes, _ = self.spike_layers[f'layer_{layer_idx}_v'].forward(V_membrane, timestep)
        
        # Reshape for multi-head attention
        Q_spikes = Q_spikes.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K_spikes = K_spikes.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V_spikes = V_spikes.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Spike-based attention with causal mask
        # Simple spike interaction: Q * K^T
        scores = np.matmul(Q_spikes, K_spikes.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # Apply causal mask
        if causal_mask is not None:
            scores = scores + (1.0 - causal_mask) * -1e9
        
        # Convert scores to spikes through attention layer
        attn_spikes, _ = self.spike_layers[f'layer_{layer_idx}_attn'].forward(scores, timestep)
        
        # Apply attention to values
        output = np.matmul(attn_spikes, V_spikes)
        
        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
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
    
    def forward(self, x: np.ndarray, use_cache: bool = False) -> np.ndarray:
        """Forward pass for language modeling"""
        batch_size, seq_len = x.shape
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len)
        
        # Accumulate outputs over timesteps
        output_accumulator = None
        
        for t in range(self.timesteps):
            # Embedding lookup
            x_embed = self.embedding[x]
            x_embed = x_embed + self.pos_encoding[:seq_len, :]
            
            # Convert to spikes
            _, x_membrane = self.spike_layers['embedding'].forward(x_embed, t)
            
            # Pass through transformer layers
            for i, layer in enumerate(self.layers):
                # Self-attention with causal mask
                residual = x_membrane
                attn_output = self.spike_driven_self_attention(
                    x_membrane, layer, i, causal_mask, t
                )
                x_membrane = residual + attn_output
                x_membrane = self.layer_norm(x_membrane, layer['ln1_gamma'], layer['ln1_beta'])
                
                # Feed-forward
                residual = x_membrane
                ff_output = self.spike_driven_feed_forward(x_membrane, layer, i, t)
                x_membrane = residual + ff_output
                x_membrane = self.layer_norm(x_membrane, layer['ln2_gamma'], layer['ln2_beta'])
            
            # Accumulate outputs
            if output_accumulator is None:
                output_accumulator = x_membrane
            else:
                output_accumulator += x_membrane
        
        # Average over timesteps
        output_avg = output_accumulator / self.timesteps
        
        # Project to vocabulary
        logits = np.matmul(output_avg, self.W_out) + self.b_out
        
        return logits
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> str:
        """Generate text from a prompt"""
        # Tokenize prompt
        tokens = [self.char_to_idx.get(c, self.char_to_idx['<PAD>']) for c in prompt]
        tokens = [self.char_to_idx['<START>']] + tokens
        
        generated = tokens.copy()
        
        for _ in range(max_length):
            # Get current sequence (truncate if too long)
            current_seq = generated[-self.max_seq_len:]
            x = np.array([current_seq])
            
            # Forward pass
            logits = self.forward(x)
            
            # Get next token probabilities
            next_token_logits = logits[0, len(current_seq)-1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Sample next token
            probs = self.softmax(next_token_logits)
            next_token = np.random.choice(self.vocab_size, p=probs)
            
            # Stop if END token
            if next_token == self.char_to_idx['<END>']:
                break
            
            generated.append(next_token)
        
        # Convert back to text
        text = ''.join([self.idx_to_char[t] for t in generated[1:] if t not in [0, 1, 2]])
        return text
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def train_on_text(self, texts: List[str], epochs: int = 100, batch_size: int = 8):
        """Train the model on text data"""
        print(f"Training Spike-LLM on {len(texts)} texts")
        print(f"Architecture: {self.n_layers} layers, {self.n_heads} heads, {self.d_model} dimensions")
        print("="*60)
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            # Shuffle texts
            random.shuffle(texts)
            
            for text in texts:
                # Prepare sequence with START and END tokens
                tokens = [self.char_to_idx['<START>']]
                tokens += [self.char_to_idx.get(c, self.char_to_idx['<PAD>']) for c in text[:self.max_seq_len-2]]
                tokens += [self.char_to_idx['<END>']]
                
                # Pad if necessary
                while len(tokens) < self.max_seq_len:
                    tokens.append(self.char_to_idx['<PAD>'])
                
                # Create input (all but last) and target (all but first)
                x = np.array([tokens[:-1]])
                y = np.array(tokens[1:])
                
                # Forward pass
                logits = self.forward(x)
                
                # Compute loss (only on non-padding tokens)
                loss = 0
                for i in range(len(y)):
                    if y[i] != self.char_to_idx['<PAD>']:
                        probs = self.softmax(logits[0, i, :])
                        loss -= np.log(probs[y[i]] + 1e-10)
                
                total_loss += loss
                n_batches += 1
                
                # Backward pass (simplified - you can adapt your full backward method)
                self.simple_backward(x, y, logits)
            
            if epoch % LOG_INTERVAL == 0:
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
                
                # Generate sample
                sample = self.generate("The ", max_length=50)
                print(f"Sample: {sample[:100]}...")
    
    def simple_backward(self, x: np.ndarray, y: np.ndarray, logits: np.ndarray):
        """Simplified backward pass for language modeling"""
        self.adam_t += 1
        batch_size, seq_len = x.shape
        
        # Compute gradients for output layer
        grad_logits = np.zeros_like(logits)
        for i in range(seq_len):
            if y[i] != self.char_to_idx['<PAD>']:
                probs = self.softmax(logits[0, i, :])
                grad_logits[0, i, :] = probs
                grad_logits[0, i, y[i]] -= 1.0
        
        # This is simplified - in practice you'd propagate through all layers
        # For now, just update output layer
        output_features = self.forward(x)[:, :, :self.d_model]  # Get features before projection
        grad_W_out = np.matmul(output_features.reshape(-1, self.d_model).T, 
                              grad_logits.reshape(-1, self.vocab_size))
        grad_b_out = np.sum(grad_logits, axis=(0, 1), keepdims=True)
        
        # Adam update
        self.W_out = self.adam_update('W_out', self.W_out, grad_W_out)
        self.b_out = self.adam_update('b_out', self.b_out, grad_b_out)
    
    def adam_update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        grad = np.clip(grad, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE)
        
        beta1, beta2 = ADAM_BETA1, ADAM_BETA2
        eps = ADAM_EPSILON
        
        self.adam_m[param_name] = beta1 * self.adam_m[param_name] + (1 - beta1) * grad
        self.adam_v[param_name] = beta2 * self.adam_v[param_name] + (1 - beta2) * grad**2
        
        m_hat = self.adam_m[param_name] / (1 - beta1**self.adam_t)
        v_hat = self.adam_v[param_name] / (1 - beta2**self.adam_t)
        
        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)


# Example usage
if __name__ == "__main__":
    # Create model
    model = SpikeLLM(
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        timesteps=TIMESTEPS,
        learning_rate=LEARNING_RATE
    )
    
    # Example training data (replace with your Wikipedia/papers dataset)
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neurons communicate through electrical and chemical signals.",
        "Wikipedia is a free online encyclopedia.",
        "Transformers revolutionized natural language processing.",
        "Spiking neural networks mimic biological neurons.",
        "The human brain contains billions of neurons.",
        "Deep learning uses multiple layers of processing.",
    ]
    
    # Train model
    model.train_on_text(training_texts, epochs=50)
    
    # Generate some text
    print("\n" + "="*60)
    print("Text Generation Examples:")
    print("="*60)
    
    prompts = ["The ", "Machine ", "Spike ", "Neural "]
    for prompt in prompts:
        generated = model.generate(prompt, max_length=100, temperature=0.8)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")