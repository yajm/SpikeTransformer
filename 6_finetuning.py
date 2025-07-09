import json
from typing import List, Optional
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

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for MLX"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize LoRA matrices
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        
        # Initialize A with normal distribution and B with zeros
        self.lora_a.weight = mx.random.normal((rank, in_features)) * 0.01
        self.lora_b.weight = mx.zeros((out_features, rank))
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: x @ A @ B * scaling"""
        return (self.lora_a(x) @ self.lora_b.weight) * self.scaling

class QADataLoader:
    """Handles loading and preprocessing of Q&A pairs"""
    
    def __init__(self, qa_file: str, tokenizer, max_length: int = 256):
        self.qa_file = qa_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qa_pairs = []
        self.formatted_texts = []
        
    def load_qa_pairs(self):
        """Load Q&A pairs from JSON file"""
        print(f"Loading Q&A pairs from {self.qa_file}")
        
        with open(self.qa_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.qa_pairs = data
        print(f"Loaded {len(self.qa_pairs)} Q&A pairs")
        
        # Format Q&A pairs for training
        filtered_count = 0
        for qa in tqdm(self.qa_pairs, desc="Formatting Q&A pairs"):
            # Format the Q&A pair
            formatted = f"{qa['question']} {qa['answer']}"
            
            # Tokenize to check length
            tokens = self.tokenizer.tokenize(formatted)
            
            # Only keep if under 64 tokens
            if len(tokens) < 64:
                self.formatted_texts.append(formatted)
            else:
                filtered_count += 1
        
        print(f"Created {len(self.formatted_texts)} formatted training examples")
        print(f"Filtered out {filtered_count} examples that were >= 64 tokens")
    
    def get_training_batch(self, batch_size: int = 16):
        """Get a random batch of Q&A pairs"""
        batch = random.sample(self.formatted_texts, min(batch_size, len(self.formatted_texts)))
        return batch


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
    
    def create_training_batch(self, texts: List[str], batch_size: int = 16):
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

    def set_freeze_config(self, freeze_layers: Optional[List[int]] = None, 
                     freeze_embedding: bool = False,
                     freeze_output: bool = False):
        """Configure which layers to freeze during training
        
        Args:
            freeze_layers: List of layer indices to freeze (0-indexed)
            freeze_embedding: Whether to freeze embedding layer
            freeze_output: Whether to freeze output projection layer
        """
        self.freeze_config = {
            'layers': freeze_layers or [],
            'embedding': freeze_embedding,
            'output': freeze_output
        }
        
        # Print configuration
        trainable_layers = [i for i in range(self.n_layers) if i not in self.freeze_config['layers']]
        print(f"Freeze configuration:")
        print(f"  Frozen layers: {self.freeze_config['layers']}")
        print(f"  Trainable layers: {trainable_layers}")
        print(f"  Embedding frozen: {self.freeze_config['embedding']}")
        print(f"  Output frozen: {self.freeze_config['output']}")


    def _create_finetuning_parameter_groups(self, base_lr: float = 1e-6):
        """Create parameter groups for fine-tuning with freezing support"""
        
        # Scale learning rates for fine-tuning (generally lower than pre-training)
        lr_config = {
            'embedding': base_lr * 0.1,      # Very conservative for embeddings
            'early_layers': base_lr * 0.5,   # Lower for early layers
            'middle_layers': base_lr * 1.0,  # Standard for middle
            'final_layers': base_lr * 2.0,   # Higher for final layers
            'output_proj': base_lr * 2.5     # Highest for output projection
        }
        
        param_groups = {
            'embedding': {'lr': lr_config['embedding'], 'param_keys': []},
            'early_layers': {'lr': lr_config['early_layers'], 'param_keys': []},
            'middle_layers': {'lr': lr_config['middle_layers'], 'param_keys': []},
            'final_layers': {'lr': lr_config['final_layers'], 'param_keys': []},
            'output_proj': {'lr': lr_config['output_proj'], 'param_keys': []}
        }
        
        # Get flattened parameters
        flat_params = dict(tree_flatten(self.parameters()))
        
        # Classify parameters by layer, excluding frozen ones
        for param_key in flat_params.keys():
            # Check if embedding should be included
            if 'embedding' in param_key:
                if not self.freeze_config.get('embedding', False):
                    param_groups['embedding']['param_keys'].append(param_key)
                continue
                
            # Check if output projection should be included
            if 'out_proj' in param_key:
                if not self.freeze_config.get('output', False):
                    param_groups['output_proj']['param_keys'].append(param_key)
                continue
                
            # Check transformer layers
            if 'layer_' in param_key:
                # Extract layer number
                layer_num = int(param_key.split('layer_')[1].split('.')[0])
                
                # Skip if layer is frozen
                if layer_num in self.freeze_config.get('layers', []):
                    continue
                
                # Classify by position
                if layer_num < self.n_layers // 3:
                    param_groups['early_layers']['param_keys'].append(param_key)
                elif layer_num < 2 * self.n_layers // 3:
                    param_groups['middle_layers']['param_keys'].append(param_key)
                else:
                    param_groups['final_layers']['param_keys'].append(param_key)
        
        # Remove empty groups
        param_groups = {k: v for k, v in param_groups.items() if v['param_keys']}
        
        return param_groups


    def finetune_on_qa(self, qa_loader: QADataLoader,
                    epochs: int = 5,
                    batch_size: int = 16,
                    base_learning_rate: float = 1e-6,
                    checkpoint_dir: str = './checkpoints_qa',
                    save_interval: int = 1,
                    eval_interval: int = 100):
        """Fine-tune model on Q&A dataset with configurable layer freezing"""
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load Q&A data
        qa_loader.load_qa_pairs()
        
        # Create parameter groups for fine-tuning
        param_groups = self._create_finetuning_parameter_groups(base_learning_rate)
        
        # Create optimizers for each group
        optimizers = {}
        total_trainable_params = 0
        for group_name, group_info in param_groups.items():
            optimizers[group_name] = optim.AdamW(
                learning_rate=group_info['lr'],
                betas=[0.9, 0.999],
                eps=1e-8,
                weight_decay=0.01
            )
            total_trainable_params += len(group_info['param_keys'])
        
        # Print training configuration
        print(f"\nFine-tuning configuration:")
        print(f"Total parameters: {len(dict(tree_flatten(self.parameters())))}")
        print(f"Trainable parameters: {total_trainable_params}")
        print(f"Frozen parameters: {len(dict(tree_flatten(self.parameters()))) - total_trainable_params}")
        print(f"\nParameter groups:")
        for group_name, group_info in param_groups.items():
            print(f"  {group_name}: {group_info['lr']:.2e} ({len(group_info['param_keys'])} parameters)")
        
        # Create loss function
        def loss_fn(params, batch_x, batch_y):
            self.update(params)
            _, loss = self(batch_x, batch_y)
            return loss
        
        # Create value and grad function
        value_and_grad_fn = mx.value_and_grad(loss_fn)
        
        # Training loop
        best_loss = float('inf')
        total_steps = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            # Calculate number of batches
            n_training_samples = len(qa_loader.formatted_texts)
            n_batches_per_epoch = n_training_samples // batch_size
            
            pbar = tqdm(range(n_batches_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx in pbar:
                # Get batch
                batch_texts = qa_loader.get_training_batch(batch_size)
                batch_x, batch_y = self.create_training_batch(batch_texts, len(batch_texts))
                
                # Get current parameters
                params = self.parameters()
                
                # Forward and backward pass
                loss, grads = value_and_grad_fn(params, batch_x, batch_y)
                
                # Apply layer-wise gradient updates (only to non-frozen parameters)
                self._apply_finetuning_updates(grads, optimizers, param_groups)
                
                # Evaluate parameters
                mx.eval(self.parameters())
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                n_batches += 1
                total_steps += 1
                
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'perplexity': f'{self.compute_perplexity(loss).item():.2f}'
                })
            
            # Epoch statistics
            avg_epoch_loss = epoch_loss / n_batches
            epoch_perplexity = self.compute_perplexity(mx.array(avg_epoch_loss)).item()
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Perplexity: {epoch_perplexity:.2f}")
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.save_finetuned_checkpoint(checkpoint_dir, epoch + 1, is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_finetuned_checkpoint(checkpoint_dir, epoch + 1)
            
            # Generate sample Q&A
            print("\nSample generations:")
            self._evaluate_qa_generation()


    def _apply_finetuning_updates(self, grads, optimizers, param_groups):
        """Apply gradients only to non-frozen parameters"""
        
        # Clip gradients
        def clip_grad(g):
            if isinstance(g, mx.array):
                return mx.clip(g, -1.0, 1.0)  # Gentler clipping for fine-tuning
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
                    # Reconstruct nested structure
                    parts = param_key.split('.')
                    current = group_grads
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = flat_grads[param_key]
            
            # Apply optimizer update
            if group_grads:
                optimizers[group_name].update(self, group_grads)


    def _evaluate_qa_generation(self):
        """Generate sample Q&A to evaluate model performance"""
        test_questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "When and where was the Futurism movement founded?",
            "What does the French verb 'confire' mean?",
        ]
        
        for question in test_questions:
            generated = self.generate(question, max_length=50, temperature=0.7)
            print(f"{generated[:100]}")


    def save_finetuned_checkpoint(self, checkpoint_dir: str, epoch: int, is_best: bool = False):
        """Save fine-tuned model checkpoint with freeze configuration"""
        
        # Include freeze config in filename
        frozen_layers = len(self.freeze_config.get('layers', []))
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, f'best_qa_frozen{frozen_layers}.pkl')
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'qa_epoch{epoch}_frozen{frozen_layers}.pkl')
        
        # Convert MLX arrays to numpy
        def to_numpy(x):
            if isinstance(x, mx.array):
                return np.array(x)
            return x
        
        # Get all parameters
        params = tree_map(to_numpy, dict(tree_flatten(self.parameters())))
        
        # Save checkpoint with freeze configuration
        checkpoint = {
            'epoch': epoch,
            'model_params': params,
            'pos_encoding': to_numpy(self._pos_encoding_buffer),
            'tokenizer': self.tokenizer,
            'freeze_config': self.freeze_config,
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
        
        print(f"Saved {'best' if is_best else 'checkpoint'} to {checkpoint_path}")


    def evaluate_qa_accuracy(self, qa_loader: QADataLoader, num_samples: int = 100):
        """Evaluate model accuracy on Q&A pairs"""
        
        correct = 0
        total = 0
        
        # Sample random Q&A pairs
        sample_indices = random.sample(range(len(qa_loader.qa_pairs)), 
                                    min(num_samples, len(qa_loader.qa_pairs)))
        
        for idx in tqdm(sample_indices, desc="Evaluating Q&A accuracy"):
            qa = qa_loader.qa_pairs[idx]
            prompt = f"Question: {qa['question']} Answer:"
            
            # Generate answer
            generated = self.generate(prompt, max_length=50, temperature=0.1)  # Low temp for factual
            
            # Extract answer
            if "Answer:" in generated:
                answer = generated.split("Answer:", 1)[1].strip()
                answer = answer.split("Question:", 1)[0].strip()
                
                # Simple exact match (could be improved)
                if qa['answer'].lower() in answer.lower():
                    correct += 1
            
            total += 1
        
        accuracy = correct / total * 100
        print(f"\nQ&A Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
        return accuracy


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

    def apply_lora(self, rank: int = 8, alpha: float = 16.0, 
               target_modules: List[str] = None):
        """Apply LoRA to specified modules
        
        Args:
            rank: Rank of LoRA adaptation
            alpha: Scaling factor for LoRA
            target_modules: List of module names to apply LoRA to
                        Default: ['w_q', 'w_v'] for attention layers
        """
        if target_modules is None:
            target_modules = ['w_q', 'w_v']  # Common choice for LLMs
        
        self.lora_config = {
            'rank': rank,
            'alpha': alpha,
            'target_modules': target_modules
        }
        
        # Store original weights and create LoRA layers
        self.original_weights = {}
        self.lora_layers = {}
        
        # Apply LoRA to transformer layers
        for i in range(self.n_layers):
            layer = self.get_layer(i)
            
            for module_name in target_modules:
                if hasattr(layer, module_name):
                    original_module = getattr(layer, module_name)
                    
                    # Store original weight
                    weight_key = f'layer_{i}.{module_name}'
                    self.original_weights[weight_key] = original_module.weight
                    
                    # Create LoRA layer
                    in_features = original_module.weight.shape[1]
                    out_features = original_module.weight.shape[0]
                    lora_layer = LoRALayer(in_features, out_features, rank, alpha)
                    
                    # Store LoRA layer
                    self.lora_layers[weight_key] = lora_layer
        
        print(f"\nApplied LoRA configuration:")
        print(f"  Rank: {rank}")
        print(f"  Alpha: {alpha}")
        print(f"  Target modules: {target_modules}")
        print(f"  Total LoRA layers: {len(self.lora_layers)}")
        
        # Calculate trainable parameters
        lora_params = sum(p.size for layer in self.lora_layers.values() 
                        for p in [layer.lora_a.weight, layer.lora_b.weight])
        total_params = sum(p.size for p in tree_flatten(self.parameters())[0])
        print(f"  LoRA parameters: {lora_params:,} ({lora_params/total_params*100:.2f}% of total)")


    def forward_with_lora(self, module: nn.Linear, x: mx.array, lora_key: str) -> mx.array:
        """Forward pass with LoRA adaptation"""
        # Original forward pass
        output = module(x)
        
        # Add LoRA if exists
        if hasattr(self, 'lora_layers') and lora_key in self.lora_layers:
            lora_output = self.lora_layers[lora_key](x)
            output = output + lora_output
        
        return output


    def _create_lora_parameter_groups(self, base_lr: float = 1e-4):
        """Create parameter groups for LoRA fine-tuning"""
        
        # Only train LoRA parameters
        param_groups = {
            'lora_a': {'lr': base_lr, 'param_keys': []},
            'lora_b': {'lr': base_lr * 2.0, 'param_keys': []}  # Higher LR for B matrices
        }
        
        # Get all parameters including LoRA
        all_params = {}
        
        # Add model parameters (frozen)
        model_params = dict(tree_flatten(self.parameters()))
        all_params.update(model_params)
        
        # Add LoRA parameters (trainable)
        for key, lora_layer in self.lora_layers.items():
            all_params[f'lora_layers.{key}.lora_a.weight'] = lora_layer.lora_a.weight
            all_params[f'lora_layers.{key}.lora_b.weight'] = lora_layer.lora_b.weight
            
            param_groups['lora_a']['param_keys'].append(f'lora_layers.{key}.lora_a.weight')
            param_groups['lora_b']['param_keys'].append(f'lora_layers.{key}.lora_b.weight')
        
        return param_groups, all_params


    def finetune_with_lora(self, qa_loader: QADataLoader,
                        epochs: int = 5,
                        batch_size: int = 8,
                        base_learning_rate: float = 1e-4,
                        checkpoint_dir: str = './checkpoints_qa_lora',
                        save_interval: int = 1,
                        eval_interval: int = 100):
        """Fine-tune model using LoRA adaptation"""
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load Q&A data
        qa_loader.load_qa_pairs()
        
        # Create parameter groups for LoRA
        param_groups, all_params = self._create_lora_parameter_groups(base_learning_rate)
        
        # Create optimizers only for LoRA parameters
        optimizers = {}
        total_trainable_params = 0
        for group_name, group_info in param_groups.items():
            optimizers[group_name] = optim.AdamW(
                learning_rate=group_info['lr'],
                betas=[0.9, 0.999],
                eps=1e-8,
                weight_decay=0.01
            )
            total_trainable_params += len(group_info['param_keys'])
        
        # Print configuration
        print(f"\nLoRA Fine-tuning configuration:")
        print(f"Total parameters: {sum(p.size for p in all_params.values())}")
        print(f"Trainable LoRA parameters: {total_trainable_params}")
        print(f"Parameter groups:")
        for group_name, group_info in param_groups.items():
            print(f"  {group_name}: {group_info['lr']:.2e} ({len(group_info['param_keys'])} parameters)")
        
        # Modified forward function to use LoRA
        original_spike_attention = self.spike_driven_self_attention
        
        def spike_driven_self_attention_with_lora(x, layer, layer_idx, timestep, membrane_potentials):
            batch_size, seq_len, _ = x.shape
            
            # Generate Q, K, V with LoRA
            Q_membrane = self.forward_with_lora(layer.w_q, x, f'layer_{layer_idx}.w_q')
            K_membrane = self.forward_with_lora(layer.w_k, x, f'layer_{layer_idx}.w_k') 
            V_membrane = self.forward_with_lora(layer.w_v, x, f'layer_{layer_idx}.w_v')
            
            # Rest of the attention mechanism remains the same
            Q_spikes, Q_mem = self.get_spike_layer('q', layer_idx)(
                Q_membrane, membrane_potentials.get(f'layer_{layer_idx}_q'), timestep
            )
            K_spikes, K_mem = self.get_spike_layer('k', layer_idx)(
                K_membrane, membrane_potentials.get(f'layer_{layer_idx}_k'), timestep
            )
            V_spikes, V_mem = self.get_spike_layer('v', layer_idx)(
                V_membrane, membrane_potentials.get(f'layer_{layer_idx}_v'), timestep
            )
            
            membrane_potentials[f'layer_{layer_idx}_q'] = Q_mem
            membrane_potentials[f'layer_{layer_idx}_k'] = K_mem
            membrane_potentials[f'layer_{layer_idx}_v'] = V_mem
            
            # Continue with the rest of the original attention mechanism
            attn_mem_key = f'layer_{layer_idx}_attn'
            if attn_mem_key not in membrane_potentials or membrane_potentials[attn_mem_key] is None:
                membrane_potentials[attn_mem_key] = mx.zeros((batch_size, seq_len, self.d_model))
            
            cumulative_kv = mx.zeros((batch_size, self.d_model))
            outputs = []
            attn_membrane_full = membrane_potentials[attn_mem_key]
            
            for i in range(seq_len):
                cumulative_kv = cumulative_kv + K_spikes[:, i, :] * V_spikes[:, i, :]
                attn_output = Q_spikes[:, i, :] * cumulative_kv
                
                attn_input_expanded = mx.expand_dims(attn_output, 1)
                attn_mem_slice = mx.expand_dims(attn_membrane_full[:, i, :], 1)
                
                attn_spikes, new_attn_mem_slice = self.get_spike_layer('attn', layer_idx)(
                    attn_input_expanded, attn_mem_slice, timestep
                )
                
                attn_membrane_full = mx.concatenate([
                    attn_membrane_full[:, :i, :],
                    new_attn_mem_slice,
                    attn_membrane_full[:, i+1:, :]
                ], axis=1)
                
                outputs.append(attn_spikes[:, 0, :])
            
            membrane_potentials[attn_mem_key] = attn_membrane_full
            output = mx.stack(outputs, axis=1)
            
            # Apply LoRA to output projection if configured
            if 'w_o' in self.lora_config.get('target_modules', []):
                return self.forward_with_lora(layer.w_o, output, f'layer_{layer_idx}.w_o')
            else:
                return layer.w_o(output)
        
        # Temporarily replace the attention method
        self.spike_driven_self_attention = spike_driven_self_attention_with_lora
        
        # Create loss function that includes LoRA parameters
        def loss_fn(params, lora_params, batch_x, batch_y):
            # Update model parameters
            self.update(params)
            
            # Update LoRA parameters
            for key, value in lora_params.items():
                if 'lora_a.weight' in key:
                    layer_key = key.replace('lora_layers.', '').replace('.lora_a.weight', '')
                    self.lora_layers[layer_key].lora_a.weight = value
                elif 'lora_b.weight' in key:
                    layer_key = key.replace('lora_layers.', '').replace('.lora_b.weight', '')
                    self.lora_layers[layer_key].lora_b.weight = value
            
            _, loss = self(batch_x, batch_y)
            return loss
        
        # Create value and grad function for combined parameters
        def combined_loss_fn(all_params, batch_x, batch_y):
            # Split parameters
            model_params = {k: v for k, v in all_params.items() if not k.startswith('lora_layers')}
            lora_params = {k: v for k, v in all_params.items() if k.startswith('lora_layers')}
            return loss_fn(model_params, lora_params, batch_x, batch_y)
        
        value_and_grad_fn = mx.value_and_grad(combined_loss_fn)
        
        # Training loop
        best_loss = float('inf')
        total_steps = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            n_training_samples = len(qa_loader.formatted_texts)
            n_batches_per_epoch = n_training_samples // batch_size
            
            pbar = tqdm(range(n_batches_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx in pbar:
                batch_texts = qa_loader.get_training_batch(batch_size)
                batch_x, batch_y = self.create_training_batch(batch_texts, len(batch_texts))
                
                # Forward and backward pass
                loss, grads = value_and_grad_fn(all_params, batch_x, batch_y)
                
                # Only update LoRA parameters
                self._apply_lora_updates(grads, optimizers, param_groups, all_params)
                
                mx.eval(all_params)
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                n_batches += 1
                total_steps += 1
                
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'perplexity': f'{self.compute_perplexity(loss).item():.2f}'
                })
                
                if total_steps % eval_interval == 0:
                    self._evaluate_qa_generation()
            
            avg_epoch_loss = epoch_loss / n_batches
            epoch_perplexity = self.compute_perplexity(mx.array(avg_epoch_loss)).item()
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Perplexity: {epoch_perplexity:.2f}")
            
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.save_lora_checkpoint(checkpoint_dir, epoch + 1, is_best=True)
            
            if (epoch + 1) % save_interval == 0:
                self.save_lora_checkpoint(checkpoint_dir, epoch + 1)
            
            print("\nSample generations:")
            self._evaluate_qa_generation()
        
        # Restore original method
        self.spike_driven_self_attention = original_spike_attention


    def _apply_lora_updates(self, grads, optimizers, param_groups, all_params):
        """Apply gradients only to LoRA parameters"""
        
        # Clip gradients
        def clip_grad(g):
            if isinstance(g, mx.array):
                return mx.clip(g, -1.0, 1.0)
            return g
        
        clipped_grads = tree_map(clip_grad, grads)
        
        # Only update LoRA parameters
        for group_name, group_info in param_groups.items():
            if not group_info['param_keys']:
                continue
            
            group_grads = {}
            for param_key in group_info['param_keys']:
                if param_key in clipped_grads:
                    group_grads[param_key] = clipped_grads[param_key]
            
            if group_grads:
                # Update the all_params dict directly
                for key, grad in group_grads.items():
                    all_params[key] = all_params[key] - optimizers[group_name].learning_rate * grad


    def save_lora_checkpoint(self, checkpoint_dir: str, epoch: int, is_best: bool = False):
        """Save LoRA adapters"""
        
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, f'best_qa_lora.pkl')
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'qa_epoch{epoch}_lora.pkl')
        
        # Convert MLX arrays to numpy
        def to_numpy(x):
            if isinstance(x, mx.array):
                return np.array(x)
            return x
        
        # Save only LoRA weights
        lora_weights = {}
        for key, lora_layer in self.lora_layers.items():
            lora_weights[key] = {
                'lora_a': to_numpy(lora_layer.lora_a.weight),
                'lora_b': to_numpy(lora_layer.lora_b.weight),
                'rank': lora_layer.rank,
                'alpha': lora_layer.alpha
            }
        
        checkpoint = {
            'epoch': epoch,
            'lora_weights': lora_weights,
            'lora_config': self.lora_config,
            'base_model_path': 'checkpointsMLX/spike_llm_mlx_epoch_0.pkl'  # Reference to base model
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Saved LoRA {'best' if is_best else 'checkpoint'} to {checkpoint_path}")


    def load_lora_checkpoint(self, checkpoint_path: str):
        """Load LoRA adapters"""
        print(f"Loading LoRA checkpoint from {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Apply LoRA configuration
        config = checkpoint['lora_config']
        self.apply_lora(rank=config['rank'], alpha=config['alpha'], 
                        target_modules=config['target_modules'])
        
        # Load LoRA weights
        for key, weights in checkpoint['lora_weights'].items():
            if key in self.lora_layers:
                self.lora_layers[key].lora_a.weight = mx.array(weights['lora_a'])
                self.lora_layers[key].lora_b.weight = mx.array(weights['lora_b'])
        
        print(f"Loaded LoRA adapters from epoch {checkpoint['epoch']}")


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

def run_finetuning_experiments(model, qa_file: str = 'qa_pairs.json'):
    """Run different fine-tuning experiments"""
    
    # Experiment 1: Full fine-tuning
    print("\n=== Experiment 1: Full Fine-tuning ===")
    model.load_checkpoint(os.path.join("checkpointsMLX", "spike_llm_mlx_epoch_0.pkl"))
    
    model.set_freeze_config(freeze_layers=[], freeze_embedding=False, freeze_output=False)
    qa_loader = QADataLoader(qa_file, model.tokenizer)
    model.finetune_on_qa(qa_loader, epochs=4, base_learning_rate=1e-6, 
                        checkpoint_dir="checkpoints_qa_full")
    
    # Experiment 2: Freeze first 6 layers
    print("\n=== Experiment 2: Freeze First 6 Layers ===")
    model.load_checkpoint(os.path.join("checkpointsMLX", "spike_llm_mlx_epoch_0.pkl"))
    
    model.set_freeze_config(freeze_layers=list(range(6)), freeze_embedding=True, freeze_output=False)
    qa_loader = QADataLoader(qa_file, model.tokenizer)
    model.finetune_on_qa(qa_loader, epochs=8, base_learning_rate=2e-6,
                        checkpoint_dir="checkpoints_qa_frozen6")
    
    # Experiment 3: Freeze first 10 layers (only train last 2)
    print("\n=== Experiment 3: Freeze First 10 Layers ===")
    model.load_checkpoint(os.path.join("checkpointsMLX", "spike_llm_mlx_epoch_0.pkl"))
    
    model.set_freeze_config(freeze_layers=list(range(10)), freeze_embedding=True, freeze_output=False)
    qa_loader = QADataLoader(qa_file, model.tokenizer)
    model.finetune_on_qa(qa_loader, epochs=12, base_learning_rate=5e-6,
                        checkpoint_dir="checkpoints_qa_frozen10")

    print("\n=== Experiment 4: LoRA Fine-tuning ===")
    # Reload base model for fair comparison
    model.load_checkpoint(os.path.join("checkpointsMLX", "spike_llm_mlx_epoch_0.pkl"))
    
    # Apply LoRA with rank 8
    model.apply_lora(rank=8, alpha=16.0, target_modules=['w_q', 'w_k', 'w_v'])
    
    qa_loader = QADataLoader(qa_file, model.tokenizer)
    model.finetune_with_lora(qa_loader, epochs=5, base_learning_rate=1e-4,
                            checkpoint_dir="checkpoints_qa_lora")

if __name__ == "__main__":
        TOKENIZER_PATH = 'syllable_tokenizer.pkl'
        tokenizer = initialize_tokenizer(TOKENIZER_PATH)
        model = SpikeLLM(tokenizer=tokenizer)
        run_finetuning_experiments(model, 'qa_pairs.json')