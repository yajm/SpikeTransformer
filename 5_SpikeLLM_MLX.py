import json
from typing import List
from constants import XAVIER_MULTIPLIER, XAVIER_MUTLITPLIER_2, EMBEDDING_INIT_SCALE, EMBEDDING_INIT_BIAS, QKV_INIT_SCALE, OUTPUT_INIT_SCALE, FF_INIT_SCALE, GRADIENT_CLIP_VALUE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, SURROGATE_CLIP_VALUE, SURROGATE_BETA, SPIKE_THRESH_EMBEDDING, SPIKE_THRESH_Q, SPIKE_THRESH_K, SPIKE_THRESH_V, SPIKE_THRESH_ATTN, SPIKE_THRESH_FF1, SPIKE_THRESH_FF2, FORWARD_MEMBRANE_CLIP_THRE, LEARNING_RATE, D_MODEL, N_HEADS, N_LAYERS, D_FF, TIMESTEPS, EPOCHS, LOG_INTERVAL, DECAY_FACTOR, RESET_VALUE, MAX_SEQ_LEN, SAVE_FILEPATH, TRAIN_MODEL
import random
import os
from tqdm import tqdm
import pickle

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from tokenizer import SmartNgramTokenizer
from tokenizer2 import SyllableTokenizer

class SpikingNeuronLayer(nn.Module):
    """Leaky Integrate-and-Fire (LIF) spiking neuron layer for MLX"""
    
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
        self.decay_factor = DECAY_FACTOR
        self.reset_value = RESET_VALUE

    def __call__(self, x: mx.array, membrane_potential=None, timestep: int = 0):
        if membrane_potential is None or timestep == 0:
            membrane_potential = mx.zeros_like(x)
        
        membrane_potential = membrane_potential * self.decay_factor + x
        membrane_potential = mx.clip(membrane_potential, -FORWARD_MEMBRANE_CLIP_THRE, FORWARD_MEMBRANE_CLIP_THRE)
        spikes = (membrane_potential >= self.threshold) # .astype(mx.float16)
        new_membrane_potential = mx.where(
            spikes > 0, 
            self.reset_value, 
            membrane_potential
        )
        return spikes, new_membrane_potential

class TransformerLayer(nn.Module):
    """Single transformer layer with MLX modules"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.ff1 = nn.Linear(d_model, d_ff, bias=True)
        self.ff2 = nn.Linear(d_ff, d_model, bias=True)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)


class SpikeLLM(nn.Module):
    def __init__(self, tokenizer=None):
        super().__init__()
        
        self.d_model = D_MODEL
        self.n_heads = N_HEADS
        self.n_layers = N_LAYERS
        self.d_ff = D_FF
        self.max_seq_len = MAX_SEQ_LEN
        self.timesteps = TIMESTEPS
        self.learning_rate = LEARNING_RATE
        self.tokenizer = tokenizer
        
        # Initialize embeddings
        self.embedding = nn.Embedding(max(self.tokenizer.id_to_vocab.keys()) + 1, self.d_model)
        
        # Initialize output projection
        self.out_proj = nn.Linear(self.d_model, max(self.tokenizer.id_to_vocab.keys()) + 1, bias=True)
        
        # Initialize transformer layers - ONLY register them once
        for i in range(self.n_layers):
            layer = TransformerLayer(self.d_model, self.d_ff)
            setattr(self, f'layer_{i}', layer)
        
        # IMPORTANT: Store spike layers as regular attributes, not nn.Module attributes
        # This prevents them from being included in parameters()
        self._spike_layers = {}
        
        # Initialize spike layers but store them in a non-parameter dict
        self._spike_layers['embedding'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_EMBEDDING)
        
        # Store spike layers for each transformer layer
        for i in range(self.n_layers):
            self._spike_layers[f'{i}_q'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_Q)
            self._spike_layers[f'{i}_k'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_K)
            self._spike_layers[f'{i}_v'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_V)
            self._spike_layers[f'{i}_attn'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_ATTN)
            self._spike_layers[f'{i}_ff1'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_FF1)
            self._spike_layers[f'{i}_ff2'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_FF2)
        
        # Store positional encoding as a buffer (non-parameter)
        self._pos_encoding_buffer = None
        self._initialize_positional_encoding()
        
        self._training_history = {
            'epoch': [],
            'loss': [],
            'perplexity': []
        }
    
    def _initialize_positional_encoding(self):
        """Initialize positional encoding as a non-parameter buffer"""
        pe = self._create_positional_encoding()
        self._pos_encoding_buffer = mx.array(pe)
    
    def _create_positional_encoding(self):
        """Create sinusoidal positional encoding using MLX operations"""
        positions = mx.arange(0, self.max_seq_len, dtype=mx.float16)[:, None]
        dim_indices = mx.arange(0, self.d_model, 2, dtype=mx.float16)
        div_term = mx.exp(dim_indices * -(mx.log(10000.0) / self.d_model))
        angles = positions * div_term
        sin_embeddings = mx.sin(angles)
        cos_embeddings = mx.cos(angles)
        pe_interleaved = mx.stack([sin_embeddings, cos_embeddings], axis=2)
        pe = pe_interleaved.reshape((self.max_seq_len, self.d_model))
        return pe
    
    def get_layer(self, i):
        """Helper to get transformer layer by index"""
        return getattr(self, f'layer_{i}')

    def _xavier_init(self, n_in, n_out):
        return mx.random.normal((n_in, n_out)) * mx.sqrt(XAVIER_MULTIPLIER / n_in) * XAVIER_MUTLITPLIER_2
    
    def surrogate_gradient(self, x: mx.array, threshold: float, beta: float = SURROGATE_BETA) -> mx.array:
        centered = beta * (x - threshold)
        centered = mx.clip(centered, -SURROGATE_CLIP_VALUE, SURROGATE_CLIP_VALUE)
        sigmoid = 1 / (1 + mx.exp(-centered))
        return beta * sigmoid * (1 - sigmoid)

    def compute_loss(self, logits, target_ids):
        """Compute cross-entropy loss in FP32 for stability"""
        # Convert to FP32 for loss computation
        logits_fp32 = logits.astype(mx.float32)
        logits_flat = logits_fp32.reshape(-1, max(self.tokenizer.id_to_vocab.keys()) + 1)
        targets_flat = target_ids.reshape(-1)
        
        # Create mask for non-padding tokens
        pad_token_id = self.tokenizer.vocab_to_id['<PAD>']
        mask = (targets_flat != pad_token_id).astype(mx.float32)
        
        # Compute cross entropy in FP32
        loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Apply mask and average
        masked_loss = loss * mask
        avg_loss = mx.sum(masked_loss) / mx.maximum(mx.sum(mask), 1.0)
        
        return avg_loss

    
    def __call__(self, input_ids: mx.array, target_ids: mx.array = None):
        """Forward pass with proper membrane potential management"""
        batch_size, seq_len = input_ids.shape
        
        # Initialize membrane potentials as a local variable (not persistent across batches)
        membrane_potentials = {}
        
        # Accumulator for multi-timestep outputs
        output_accumulator = None
        
        for t in range(self.timesteps):
            # Embedding lookup
            x_embed = self.embedding(input_ids) # .astype(mx.float16)
            
            # Add positional encoding
            pos_encoding = mx.stop_gradient(self._pos_encoding_buffer[:seq_len, :]) # .astype(mx.float16)
            x_embed = x_embed + pos_encoding
            
            # Convert to spikes - USE get_spike_layer instead of direct access
            _, embed_membrane = self.get_spike_layer('embedding')(
                x_embed, 
                membrane_potentials.get('embedding'), 
                t
            )
            membrane_potentials['embedding'] = embed_membrane
            
            # Process through transformer layers
            layer_membrane = embed_membrane
            
            for i in range(self.n_layers):
                layer = self.get_layer(i)
                # Self-attention with causal mask
                layer_membrane = self.spike_transformer_block(
                    layer_membrane, layer, i, t, membrane_potentials
                )
            
            # Output projection
            if output_accumulator is None:
                output_accumulator = layer_membrane
            else:
                output_accumulator = output_accumulator + layer_membrane
        
        # Average over timesteps
        final_output = output_accumulator / self.timesteps
        
        # Project to vocabulary size
        logits = self.out_proj(final_output)
        
        if target_ids is None:
            return logits
        
        # Compute loss
        return logits, self.compute_loss(logits, target_ids)


    def encode(self, input_ids: mx.array) -> mx.array:
        """Extract embeddings with proper membrane potential management"""
        batch_size, seq_len = input_ids.shape
        
        # Initialize membrane potentials as a local variable
        membrane_potentials = {}
        
        # Accumulator for multi-timestep outputs
        output_accumulator = None
        
        for t in range(self.timesteps):
            # Embedding lookup + positional encoding
            x_embed = self.embedding(input_ids)
            pos_encoding = mx.stop_gradient(self._pos_encoding_buffer[:seq_len, :])
            x_embed = x_embed + pos_encoding
            
            # Convert to spikes
            _, embed_membrane = self.get_spike_layer('embedding')(
                x_embed, 
                membrane_potentials.get('embedding'), 
                t
            )
            membrane_potentials['embedding'] = embed_membrane
            
            # Process through transformer layers
            layer_membrane = embed_membrane
            for i in range(self.n_layers):
                layer = self.get_layer(i)
                layer_membrane = self.spike_transformer_block(
                    layer_membrane, layer, i, t, membrane_potentials
                )
            
            # Accumulate outputs
            if output_accumulator is None:
                output_accumulator = layer_membrane
            else:
                output_accumulator = output_accumulator + layer_membrane
        
        # Average over timesteps
        final_output = output_accumulator / self.timesteps
        
        # Mean pooling over sequence (ignoring padding)
        pad_token_id = self.tokenizer.vocab_to_id['<PAD>']
        embeddings = []
        
        for b in range(batch_size):
            # Create mask for non-padding tokens
            mask = (input_ids[b] != pad_token_id) # .astype(mx.float16)
            valid_positions = mx.sum(mask)
            
            if valid_positions > 0:
                # Weighted mean (only non-padding positions)
                masked_output = final_output[b] * mx.expand_dims(mask, -1)
                embedding = mx.sum(masked_output, axis=0) / valid_positions
            else:
                # Fallback if all padding
                embedding = mx.mean(final_output[b], axis=0)
            
            # L2 normalize
            embedding = embedding / (mx.linalg.norm(embedding) + 1e-8)
            embeddings.append(embedding)
        
        return mx.stack(embeddings)


    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> str:
        """Generate text with proper membrane potential management"""
        # Tokenize prompt
        tokens = self.tokenizer.tokenize(prompt)
        tokens = [self.tokenizer.vocab_to_id['<START>']] + tokens
        
        for _ in range(max_length):
            # Prepare input (pad if necessary)
            input_tokens = tokens[-self.max_seq_len:]
            if len(input_tokens) < self.max_seq_len:
                input_tokens = input_tokens + [self.tokenizer.vocab_to_id['<PAD>']] * (self.max_seq_len - len(input_tokens))
            
            input_ids = mx.array([input_tokens])
            
            # Get logits (no target_ids for generation)
            # Note: membrane potentials are created fresh for each forward pass
            logits = self(input_ids, target_ids=None)
            
            # Get prediction for the last valid position
            last_pos = min(len(tokens) - 1, self.max_seq_len - 1)
            next_logits = logits[0, last_pos, :] / temperature
            
            # Sample next token
            probs = mx.softmax(next_logits)
            next_token = mx.random.categorical(mx.log(probs))
            
            # Stop if END token
            if next_token.item() == self.tokenizer.vocab_to_id['<END>']:
                break
            
            # Add to sequence
            tokens.append(next_token.item())
        
        # Decode using tokenizer (excluding START token)
        generated_tokens = [t for t in tokens[1:] if t != self.tokenizer.vocab_to_id['<PAD>']]
        return self.tokenizer.decode(generated_tokens)

    def get_spike_layer(self, layer_type: str, layer_idx: int = None):
        """Helper to access spike layers from the non-parameter dict"""
        if layer_type == 'embedding':
            return self._spike_layers['embedding']
        else:
            return self._spike_layers[f'{layer_idx}_{layer_type}']

    def spike_transformer_block(self, x_membrane: mx.array, layer, layer_idx: int, timestep: int, membrane_potentials: dict):
        """Modified transformer block with causal masking"""
        residual = x_membrane
        
        # Self-attention
        attn_output = self.spike_driven_self_attention(
            x_membrane, layer, layer_idx, timestep, membrane_potentials
        )
        x_membrane = residual + attn_output
        x_membrane = layer.ln1(x_membrane)
        
        residual = x_membrane
        
        # Feed-forward
        ff_output = self.spike_driven_feed_forward(
            x_membrane, layer, layer_idx, timestep, membrane_potentials
        )
        x_membrane = residual + ff_output
        x_membrane = layer.ln2(x_membrane)
        
        return x_membrane

    def spike_driven_self_attention(self, x: mx.array, layer, layer_idx: int, timestep: int, membrane_potentials: dict):
        """Fixed self-attention with consistent membrane potential structure"""
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V
        Q_membrane = layer.w_q(x)
        K_membrane = layer.w_k(x)
        V_membrane = layer.w_v(x)
        
        # Convert to spikes using the registered spike layers
        Q_spikes, Q_mem = self.get_spike_layer('q', layer_idx)(
            Q_membrane, 
            membrane_potentials.get(f'layer_{layer_idx}_q'), 
            timestep
        )
        K_spikes, K_mem = self.get_spike_layer('k', layer_idx)(
            K_membrane,
            membrane_potentials.get(f'layer_{layer_idx}_k'),
            timestep
        )
        V_spikes, V_mem = self.get_spike_layer('v', layer_idx)(
            V_membrane,
            membrane_potentials.get(f'layer_{layer_idx}_v'),
            timestep
        )
        
        # Update membrane potentials
        membrane_potentials[f'layer_{layer_idx}_q'] = Q_mem
        membrane_potentials[f'layer_{layer_idx}_k'] = K_mem
        membrane_potentials[f'layer_{layer_idx}_v'] = V_mem
        
        # Initialize attention membrane potential if not exists (fixed size for all positions)
        attn_mem_key = f'layer_{layer_idx}_attn'
        if attn_mem_key not in membrane_potentials or membrane_potentials[attn_mem_key] is None:
            membrane_potentials[attn_mem_key] = mx.zeros((batch_size, seq_len, self.d_model))
        
        # Linear attention with causal masking
        cumulative_kv = mx.zeros((batch_size, self.d_model))
        outputs = []
        
        # Get the full attention membrane potential tensor
        attn_membrane_full = membrane_potentials[attn_mem_key]
        
        for i in range(seq_len):
            cumulative_kv = cumulative_kv + K_spikes[:, i, :] * V_spikes[:, i, :]
            attn_output = Q_spikes[:, i, :] * cumulative_kv
            
            # Process spike for this position using the shared membrane potential
            attn_input_expanded = mx.expand_dims(attn_output, 1)
            attn_mem_slice = mx.expand_dims(attn_membrane_full[:, i, :], 1)
            
            attn_spikes, new_attn_mem_slice = self.get_spike_layer('attn', layer_idx)(
                attn_input_expanded,
                attn_mem_slice,
                timestep
            )
            
            # Update the membrane potential for this position
            attn_membrane_full = mx.concatenate([
                attn_membrane_full[:, :i, :],
                new_attn_mem_slice,
                attn_membrane_full[:, i+1:, :]
            ], axis=1)
            
            outputs.append(attn_spikes[:, 0, :])
        
        # Update the full attention membrane potential
        membrane_potentials[attn_mem_key] = attn_membrane_full
        
        output = mx.stack(outputs, axis=1)
        return layer.w_o(output)

    def spike_driven_feed_forward(self, x: mx.array, layer, layer_idx: int, timestep: int, membrane_potentials: dict):
        """Feed-forward network"""
        hidden_membrane = layer.ff1(x)
        hidden_spikes, hidden_mem = self.get_spike_layer('ff1', layer_idx)(
            hidden_membrane,
            membrane_potentials.get(f'layer_{layer_idx}_ff1'),
            timestep
        )
        membrane_potentials[f'layer_{layer_idx}_ff1'] = hidden_mem
        
        # Second layer
        output_membrane = layer.ff2(hidden_spikes)
        _, output_mem_after = self.get_spike_layer('ff2', layer_idx)(
            output_membrane,
            membrane_potentials.get(f'layer_{layer_idx}_ff2'),
            timestep
        )
        membrane_potentials[f'layer_{layer_idx}_ff2'] = output_mem_after
        
        return output_mem_after

    def get_text_embedding(self, text: str) -> mx.array:
        """Get embedding for a single text"""
        # Tokenize
        tokens = [self.tokenizer.vocab_to_id['<START>']]
        tokens += self.preprocess_text(text[:self.max_seq_len-2])
        tokens += [self.tokenizer.vocab_to_id['<END>']]
        
        # Pad
        while len(tokens) < self.max_seq_len:
            tokens.append(self.tokenizer.vocab_to_id['<PAD>'])
        tokens = tokens[:self.max_seq_len]
        
        # Get embedding
        input_ids = mx.array([tokens])
        embedding = self.encode(input_ids)
        
        return embedding[0]

    def preprocess_text(self, text: str) -> List[int]:
        """Convert text to token indices using the syllable tokenizer"""
        return self.tokenizer.tokenize(text)
    
    def create_training_batch(self, texts: List[str], batch_size: int = 8):
        """Create batches of training data using the tokenizer"""
        batch_x = []
        batch_y = []
        
        for text in texts[:batch_size]:
            # Tokenize with START and END tokens
            tokens = [self.tokenizer.vocab_to_id['<START>']]
            tokens += self.preprocess_text(text[:self.max_seq_len-2])
            tokens += [self.tokenizer.vocab_to_id['<END>']]
            
            # Pad to max_seq_len
            while len(tokens) < self.max_seq_len:
                tokens.append(self.tokenizer.vocab_to_id['<PAD>'])
            
            # Truncate if necessary
            tokens = tokens[:self.max_seq_len]
            
            # Create input (all but last) and target (all but first)
            batch_x.append(tokens[:-1])
            batch_y.append(tokens[1:])
        
        return mx.array(batch_x), mx.array(batch_y)
    
    def compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from loss"""
        return mx.exp(mx.minimum(loss, 100.0))
    
  
    def train_on_wikipedia(self, data_loader, 
                    epochs: int = 10, 
                    batch_size: int = 8,
                    chunks_per_epoch: int = 1000,
                    checkpoint_dir: str = './checkpoints',
                    save_interval: int = 20):
        """Train on Wikipedia dataset with layer-wise learning rates"""
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Define layer-wise learning rates
        base_lr = self.learning_rate  # 1e-5
        lr_config = {
            'embedding': base_lr * 0.3,      # 3e-6 - even more conservative
            'early_layers': base_lr * 0.7,   # 7e-6 - reduced due to temporal amplification  
            'middle_layers': base_lr * 1.5,  # 1.5e-5 - moderate
            'final_layers': base_lr * 3.5,   # 3.5e-5 - aggressive for plateau breaking
            'output_proj': base_lr * 4.0     # 4e-5 - most aggressive
        }
        
        # Group parameters by layer type
        param_groups = self._create_parameter_groups(lr_config)
        
        # Create optimizers for each group
        optimizers = {}
        for group_name, group_info in param_groups.items():
            optimizers[group_name] = optim.AdamW(
                learning_rate=group_info['lr'],
                betas=[ADAM_BETA1, ADAM_BETA2],
                eps=ADAM_EPSILON,
                weight_decay=0.01
            )
        
        # Create the loss function
        def loss_fn(params, batch_x, batch_y):
            self.update(params)
            _, loss = self(batch_x, batch_y)
            return loss
        
        # Create value and grad function
        value_and_grad_fn = mx.value_and_grad(loss_fn)
        
        print(f"Starting training with layer-wise learning rates:")
        for group_name, group_info in param_groups.items():
            print(f"  {group_name}: {group_info['lr']:.2e} ({len(group_info['param_keys'])} parameters)")
        
        for epoch in range(epochs):
            epoch_chunks = data_loader.get_training_chunks(chunks_per_epoch)
            random.shuffle(epoch_chunks)
            
            epoch_loss = 0
            n_batches = 0
            
            pbar = tqdm(range(0, len(epoch_chunks), batch_size), 
                    desc=f"Epoch {epoch+1}/{epochs}")
            
            for i in pbar:
                batch_texts = epoch_chunks[i:i + batch_size]
                batch_x, batch_y = self.create_training_batch(batch_texts, len(batch_texts))
                
                # Get current parameters
                params = self.parameters()
                
                # Forward and backward pass
                loss, grads = value_and_grad_fn(params, batch_x, batch_y)
                
                # Apply layer-wise gradient updates
                self._apply_layerwise_updates(grads, optimizers, param_groups)
                
                # Evaluate parameters
                mx.eval(self.parameters())
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                n_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'perplexity': f'{self.compute_perplexity(loss).item():.2f}'
                })
            
            # Rest of the epoch handling remains the same...
            avg_epoch_loss = epoch_loss / n_batches
            epoch_perplexity = self.compute_perplexity(mx.array(avg_epoch_loss)).item()
            
            self._training_history['epoch'].append(epoch + 1)
            self._training_history['loss'].append(avg_epoch_loss)
            self._training_history['perplexity'].append(epoch_perplexity)
            
            print(f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Perplexity: {epoch_perplexity:.2f}")
            
            if (epoch + 1) % LOG_INTERVAL == 0:
                prompts = ["The capital of France is", "What", "Elon Musk is a", "When was"]
                for prompt in prompts:
                    generated = self.generate(prompt, max_length=100, temperature=0.8)
                    print(f"Sample '{prompt}': {generated[:200]}")
            
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(checkpoint_dir, epoch + 1)

    def _create_parameter_groups(self, lr_config):
        """Create parameter groups for layer-wise learning rates"""
        param_groups = {
            'embedding': {'lr': lr_config['embedding'], 'param_keys': []},
            'early_layers': {'lr': lr_config['early_layers'], 'param_keys': []},
            'middle_layers': {'lr': lr_config['middle_layers'], 'param_keys': []},
            'final_layers': {'lr': lr_config['final_layers'], 'param_keys': []},
            'output_proj': {'lr': lr_config['output_proj'], 'param_keys': []}
        }
        
        # Get flattened parameters to understand the structure
        flat_params = dict(tree_flatten(self.parameters()))
        
        # Classify parameters by layer
        for param_key in flat_params.keys():
            if 'embedding' in param_key:
                param_groups['embedding']['param_keys'].append(param_key)
            elif 'out_proj' in param_key:
                param_groups['output_proj']['param_keys'].append(param_key)
            elif 'layer_' in param_key:
                # Extract layer number
                layer_num = int(param_key.split('layer_')[1].split('.')[0])
                
                if layer_num < self.n_layers // 3:  # First third
                    param_groups['early_layers']['param_keys'].append(param_key)
                elif layer_num < 2 * self.n_layers // 3:  # Middle third
                    param_groups['middle_layers']['param_keys'].append(param_key)
                else:  # Final third
                    param_groups['final_layers']['param_keys'].append(param_key)
            else:
                # Default to middle layers for any unclassified parameters
                param_groups['middle_layers']['param_keys'].append(param_key)
        
        return param_groups

    def _apply_layerwise_updates(self, grads, optimizers, param_groups):
        """Apply gradients with different learning rates per layer group"""
        
        # Clip gradients globally first
        def clip_grad(g):
            if isinstance(g, mx.array):
                return mx.clip(g, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE)
            return g
        
        clipped_grads = tree_map(clip_grad, grads)
        flat_grads = dict(tree_flatten(clipped_grads))
        
        # Apply updates for each parameter group
        for group_name, group_info in param_groups.items():
            if not group_info['param_keys']:
                continue
                
            # Create gradients dict for this group only
            group_grads = {}
            for param_key in group_info['param_keys']:
                if param_key in flat_grads:
                    # Reconstruct nested structure for this parameter
                    parts = param_key.split('.')
                    current = group_grads
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = flat_grads[param_key]
            
            # Apply optimizer update for this group
            if group_grads:  # Only update if we have gradients for this group
                optimizers[group_name].update(self, group_grads)

    # Add these methods to your SpikeLLM class
    
    def save_checkpoint(self, checkpoint_dir: str, epoch: int):
        """Save model checkpoint with MLX arrays converted to numpy for pickling"""
        checkpoint_path = os.path.join(checkpoint_dir, f'spike_llm_mlx_epoch_{epoch}.pkl')
        
        # Convert MLX arrays to numpy for saving
        def to_numpy(x):
            if isinstance(x, mx.array):
                return np.array(x)
            return x
        
        # Get all parameters
        params = tree_map(to_numpy, dict(tree_flatten(self.parameters())))
        
        # Save positional encoding buffer separately
        checkpoint = {
            'epoch': epoch,
            'model_params': params,
            'pos_encoding': to_numpy(self._pos_encoding_buffer),
            'tokenizer': self.tokenizer,
            'training_history': self._training_history,
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
        
        if 'tokenizer' in checkpoint:
            self.tokenizer = checkpoint['tokenizer']
        
        # Convert numpy arrays back to MLX arrays
        def to_mlx(x):
            if isinstance(x, np.ndarray):
                return mx.array(x)
            return x
        
        # Load parameters - need to convert from flat to nested structure
        flat_params = tree_map(to_mlx, checkpoint['model_params'])
        
        # Convert flat dictionary to nested structure
        nested_params = {}
        for key, value in flat_params.items():
            parts = key.split('.')
            current = nested_params
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        
        # Update model parameters
        self.update(nested_params)
        
        # Restore positional encoding buffer
        if 'pos_encoding' in checkpoint:
            self._pos_encoding_buffer = mx.array(checkpoint['pos_encoding'])
        
        # Restore training history
        self._training_history = checkpoint['training_history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

# WikipediaDataLoader class remains the same...
class WikipediaDataLoader:
    """Handles loading and preprocessing of Wikipedia articles"""
    
    def __init__(self, data_folder: str, max_article_length: int = 10000, 
                 chunk_size: int = 512, overlap: int = 64, tokenizer=None):
        self.data_folder = data_folder
        self.max_article_length = max_article_length
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.articles = []
        self.chunks = []
        self.tokenizer = tokenizer
        
    def load_articles(self):
        """Load articles from all JSON files in the folder"""
        
        json_files = [f for f in os.listdir(self.data_folder) if f.endswith('.json')]
        print(f"Loading Wikipedia articles from {len(json_files)} files in {self.data_folder}")
        
        total_articles = 0
        for json_file in tqdm(json_files, desc="Loading files"):
            file_path = os.path.join(self.data_folder, json_file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract content from each article
            for article in data:
                if 'content' in article and article['content']:
                    # Take only first max_article_length characters to manage memory
                    content = article['content'][:self.max_article_length]
                    self.articles.append({
                        'title': article.get('title', 'Unknown'),
                        'content': content
                    })
                    total_articles += 1
        
        print(f"Loaded {total_articles} articles from {len(json_files)} files")
        
    def create_chunks(self):
        """Split articles into overlapping chunks for training, filtering out chunks with UNK tokens"""
        print("Creating training chunks...")
        
        if self.tokenizer is None:
            print("Warning: No tokenizer provided, cannot filter UNK tokens during chunk creation")
            
        total_chunks = 0
        filtered_chunks = 0
        total_unk_count = 0
        
        for article in tqdm(self.articles, desc="Chunking articles"):
            content = article['content']
            
            # Skip very short articles
            if len(content) < 100:
                continue
                
            # Create overlapping chunks
            for i in range(0, len(content) - self.chunk_size + 1, self.chunk_size - self.overlap):
                chunk = content[i:i + self.chunk_size]
                if len(chunk) >= 100:  # Minimum chunk size
                    total_chunks += 1
                    
                    # Clean the chunk before adding it
                    if self.tokenizer is not None:
                        # Get valid characters from tokenizer vocabulary
                        tokens = self.tokenizer.tokenize(chunk)
                        
                        total_unk_count += tokens.count(3)
                        
                        valid_chars = set()
                        for token in self.tokenizer.vocab:
                            if len(token) == 1:  # Single character tokens
                                valid_chars.add(token)
                        
                        # Always keep space and newline
                        valid_chars.add(' ')
                        valid_chars.add('\n')
                        
                        # Clean the chunk - remove invalid characters
                        cleaned_chunk = ''.join(char for char in chunk if char.lower() in valid_chars or char.isupper() and char.lower() in valid_chars)
                        
                        # Skip if chunk becomes too short after cleaning
                        if len(cleaned_chunk.strip()) < 50:
                            filtered_chunks += 1
                            continue
                        
                        # Add the CLEANED chunk
                        self.chunks.append(cleaned_chunk)
                    else:
                        # No tokenizer, add raw chunk
                        self.chunks.append(chunk)

        print("Total Unk count", total_unk_count)
        print(f"Created {len(self.chunks)} training chunks")
        if self.tokenizer is not None:
            print(f"Filtered out {filtered_chunks} chunks that were too short after cleaning ({filtered_chunks/total_chunks*100:.1f}%)")
            
    def get_training_chunks(self, num_chunks: int = None):
        """Get training chunks, optionally limiting the number"""
        if num_chunks:
            return random.sample(self.chunks, min(num_chunks, len(self.chunks)))
        return self.chunks

def initialize_tokenizer(tokenizer_path: str = None):
    print(f"Loading tokenizer from {tokenizer_path}")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    print("=== VOCABULARY CHECK ===")
    print(f"'<UNK>' in vocab_to_id: {'<UNK>' in tokenizer.vocab_to_id}")
    print(f"UNK in any vocab tokens: {any('UNK' in token for token in tokenizer.vocab)}")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"First 20 tokens: {tokenizer.vocab[:20]}")

    # Check if any ID maps to UNK
    unk_ids = [id for id, token in tokenizer.id_to_vocab.items() if 'UNK' in str(token)]
    print(f"IDs that map to UNK: {unk_ids}")
    
    return tokenizer

if __name__ == "__main__":
    WIKIPEDIA_DATA_FOLDER = "scraped_data"
    TOKENIZER_PATH = 'syllable_tokenizer.pkl'
    
    if TRAIN_MODEL:
        tokenizer = initialize_tokenizer(TOKENIZER_PATH)
        data_loader = WikipediaDataLoader(
            data_folder=WIKIPEDIA_DATA_FOLDER,
            max_article_length=100000,
            chunk_size=256,
            overlap=64,
            tokenizer=tokenizer
        )
        data_loader.load_articles()
        data_loader.create_chunks()


        model = SpikeLLM(tokenizer=tokenizer)
 
        model.load_checkpoint(os.path.join("checkpointsMLX", "spike_llm_mlx_epoch_0.pkl"))
        print(f"Tokenizer vocab size: {len(model.tokenizer.vocab)}")
        print(f"Tokenizer max ID: {max(model.tokenizer.id_to_vocab.keys())}")
        print(f"Embedding weight shape: {model.embedding.weight.shape}")
        model.train_on_wikipedia(
            data_loader=data_loader,
            epochs=EPOCHS,
            batch_size=16,
            chunks_per_epoch=2048,
            checkpoint_dir="checkpointsMLX",
            save_interval=10
        )
        # Test embeddings
        embedding1 = model.get_text_embedding("The capital of France")
        embedding2 = model.get_text_embedding("Paris is a city") 
        similarity = mx.sum(embedding1 * embedding2).item()
        print(f"Similarity: {similarity}")
    else:
        # Load and test model
        model = SpikeLLM(tokenizer_path='tokenizer.pkl')
        checkpoint_path = os.path.join(SAVE_FILEPATH, 'spike_llm_mlx_epoch_10.pkl')
        if os.path.exists(checkpoint_path):
            model.load_checkpoint(checkpoint_path)
            
            # Generate samples
            prompts = ["The ", "In the ", "Wikipedia ", "A "]
            for prompt in prompts:
                generated = model.generate(prompt, max_length=200, temperature=0.8)
                print(f"\nPrompt: '{prompt}'")
                print(f"Generated: {generated}")
                
                # Show tokenization analysis
                analysis = model.tokenizer.analyze_compression(generated)
                print(f"Compression ratio: {analysis['compression_ratio']:.2f}x")
                print(f"Space savings: {analysis['savings_percentage']:.1f}%")