import json
from typing import List
from constants import XAVIER_MULTIPLIER, XAVIER_MUTLITPLIER_2, EMBEDDING_INIT_SCALE, EMBEDDING_INIT_BIAS, QKV_INIT_SCALE, OUTPUT_INIT_SCALE, FF_INIT_SCALE, GRADIENT_CLIP_VALUE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, SURROGATE_CLIP_VALUE, SURROGATE_BETA, SPIKE_THRESH_EMBEDDING, SPIKE_THRESH_Q, SPIKE_THRESH_K, SPIKE_THRESH_V, SPIKE_THRESH_ATTN, SPIKE_THRESH_FF1, SPIKE_THRESH_FF2, FORWARD_MEMBRANE_CLIP_THRE, LEARNING_RATE, D_MODEL, N_HEADS, N_LAYERS, D_FF, TIMESTEPS, EPOCHS, LOG_INTERVAL, DECAY_FACTOR, RESET_VALUE, MAX_SEQ_LEN, SAVE_FILEPATH, TRAIN_MODEL
import random
import os
from tqdm import tqdm
import pickle
import os

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from tokenizer2 import SyllableTokenizer

class SpikingNeuronLayer(nn.Module):
    """Leaky Integrate-and-Fire (LIF) spiking neuron layer for MLX"""
    
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
        self.decay_factor = DECAY_FACTOR
        self.reset_value = RESET_VALUE
        self.membrane_potential = None

    def __call__(self, x: mx.array, timestep: int = 0):
        if self.membrane_potential is None or timestep == 0:
            self.membrane_potential = mx.zeros_like(x)
        
        self.membrane_potential = self.membrane_potential * self.decay_factor + x
        self.membrane_potential = mx.clip(self.membrane_potential, -FORWARD_MEMBRANE_CLIP_THRE, FORWARD_MEMBRANE_CLIP_THRE)
        spikes = (self.membrane_potential >= self.threshold).astype(mx.float32)
        self.membrane_potential = mx.where(
            spikes > 0, 
            self.reset_value, 
            self.membrane_potential
        )
        return spikes, self.membrane_potential

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
        self.vocab = self.tokenizer.vocab
        self.vocab_size = max(self.tokenizer.id_to_vocab.keys()) + 1
        self.token_to_idx = self.tokenizer.vocab_to_id
        self.idx_to_token = self.tokenizer.id_to_vocab
        
        # Initialize embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Initialize output projection
        self.out_proj = nn.Linear(self.d_model, self.vocab_size, bias=True)
        
        # Initialize transformer layers as proper MLX modules
        self.transformer_layers = [TransformerLayer(self.d_model, self.d_ff) for _ in range(self.n_layers)]
        
        # Initialize spike layers
        self.spike_layers = {}
        self.spike_layers['embedding'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_EMBEDDING)
        
        for i in range(self.n_layers):
            self.spike_layers[f'layer_{i}_q'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_Q)
            self.spike_layers[f'layer_{i}_k'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_K)
            self.spike_layers[f'layer_{i}_v'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_V)
            self.spike_layers[f'layer_{i}_attn'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_ATTN)
            self.spike_layers[f'layer_{i}_ff1'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_FF1)
            self.spike_layers[f'layer_{i}_ff2'] = SpikingNeuronLayer(threshold=SPIKE_THRESH_FF2)
        
        # Positional encoding as a frozen parameter
        pe = self._create_positional_encoding()
        self.pos_encoding = mx.array(pe)
        
        # Apply custom initialization
        self._init_weights()
        
        self.training_history = {
            'epoch': [],
            'loss': [],
            'perplexity': []
        }

    def _create_positional_encoding(self):
        """Create sinusoidal positional encoding"""
        pe = np.zeros((self.max_seq_len, self.d_model))
        position = np.arange(0, self.max_seq_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(10 / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def _init_weights(self):
        """Custom weight initialization"""
        # Embedding initialization
        self.embedding.weight = self._xavier_init(self.vocab_size, self.d_model) * EMBEDDING_INIT_SCALE
        self.embedding.weight = self.embedding.weight + EMBEDDING_INIT_BIAS
        
        # Output projection initialization
        self.out_proj.weight = self._xavier_init(self.d_model, self.vocab_size) * OUTPUT_INIT_SCALE
        
        # Layer initialization
        for layer in self.transformer_layers:
            layer.w_q.weight = self._xavier_init(self.d_model, self.d_model) * QKV_INIT_SCALE
            layer.w_k.weight = self._xavier_init(self.d_model, self.d_model) * QKV_INIT_SCALE
            layer.w_v.weight = self._xavier_init(self.d_model, self.d_model) * QKV_INIT_SCALE
            layer.w_o.weight = self._xavier_init(self.d_model, self.d_model) * OUTPUT_INIT_SCALE
            layer.ff1.weight = self._xavier_init(self.d_model, self.d_ff) * FF_INIT_SCALE
            layer.ff2.weight = self._xavier_init(self.d_ff, self.d_model) * FF_INIT_SCALE

    def _xavier_init(self, n_in, n_out):
        return mx.random.normal((n_in, n_out)) * mx.sqrt(XAVIER_MULTIPLIER / n_in) * XAVIER_MUTLITPLIER_2
    
    def surrogate_gradient(self, x: mx.array, threshold: float, beta: float = SURROGATE_BETA) -> mx.array:
        centered = beta * (x - threshold)
        centered = mx.clip(centered, -SURROGATE_CLIP_VALUE, SURROGATE_CLIP_VALUE)
        sigmoid = 1 / (1 + mx.exp(-centered))
        return beta * sigmoid * (1 - sigmoid)

    def compute_loss(self, logits, target_ids):
        """Compute cross-entropy loss"""
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_ids.reshape(-1)
        
        # Create mask for non-padding tokens
        pad_token_id = self.token_to_idx['<PAD>']
        mask = (targets_flat != pad_token_id).astype(mx.float32)
        
        # Compute cross entropy
        loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Apply mask and average
        masked_loss = loss * mask
        avg_loss = mx.sum(masked_loss) / mx.maximum(mx.sum(mask), 1.0)
        
        return avg_loss

    def reset_spike_states(self):
        """Reset all membrane potentials"""
        for name, layer in self.spike_layers.items():
            layer.membrane_potential = None

    def __call__(self, input_ids: mx.array, target_ids: mx.array = None):
        """Forward pass for LLM"""
        self.reset_spike_states()
        batch_size, seq_len = input_ids.shape
        
        # Accumulator for multi-timestep outputs
        output_accumulator = None
        
        for t in range(self.timesteps):
            # Embedding lookup
            x_embed = self.embedding(input_ids)
            
            # Add positional encoding
            x_embed = x_embed + self.pos_encoding[:seq_len, :]
            
            # Convert to spikes
            _, embed_membrane = self.spike_layers['embedding'](x_embed, t)
            
            # Process through transformer layers
            layer_membrane = embed_membrane
            
            for i, layer in enumerate(self.transformer_layers):
                # Self-attention with causal mask
                layer_membrane = self.spike_transformer_block(
                    layer_membrane, layer, i, t
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

    def spike_transformer_block(self, x_membrane: mx.array, layer, layer_idx: int, timestep: int):
        """Modified transformer block with causal masking"""
        residual = x_membrane
        
        # Self-attention
        attn_output = self.spike_driven_self_attention(
            x_membrane, layer, layer_idx, timestep
        )
        x_membrane = residual + attn_output
        x_membrane = layer.ln1(x_membrane)
        
        residual = x_membrane
        
        # Feed-forward
        ff_output = self.spike_driven_feed_forward(
            x_membrane, layer, layer_idx, timestep
        )
        x_membrane = residual + ff_output
        x_membrane = layer.ln2(x_membrane)
        
        return x_membrane

    def spike_driven_self_attention(self, x: mx.array, layer, layer_idx: int, timestep: int):
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V
        Q_membrane = layer.w_q(x)
        K_membrane = layer.w_k(x)
        V_membrane = layer.w_v(x)
        
        # Convert to spikes
        Q_spikes, _ = self.spike_layers[f'layer_{layer_idx}_q'](Q_membrane, timestep)
        K_spikes, _ = self.spike_layers[f'layer_{layer_idx}_k'](K_membrane, timestep)
        V_spikes, _ = self.spike_layers[f'layer_{layer_idx}_v'](V_membrane, timestep)
        
        # Linear attention with causal masking
        cumulative_kv = mx.zeros((batch_size, self.d_model))
        outputs = []
        
        for i in range(seq_len):
            cumulative_kv = cumulative_kv + K_spikes[:, i, :] * V_spikes[:, i, :]
            attn_output = Q_spikes[:, i, :] * cumulative_kv
            attn_spikes, _ = self.spike_layers[f'layer_{layer_idx}_attn'](
                mx.expand_dims(attn_output, 1), timestep
            )
            outputs.append(attn_spikes[:, 0, :])
        
        output = mx.stack(outputs, axis=1)
        return layer.w_o(output)

    def spike_driven_feed_forward(self, x: mx.array, layer, layer_idx: int, timestep: int):
        """Feed-forward network"""
        # First layer
        hidden_membrane = layer.ff1(x)
        hidden_spikes, _ = self.spike_layers[f'layer_{layer_idx}_ff1'](hidden_membrane, timestep)
        
        # Second layer
        output_membrane = layer.ff2(hidden_spikes)
        _, output_mem_after = self.spike_layers[f'layer_{layer_idx}_ff2'](output_membrane, timestep)
        
        return output_mem_after
    
    def encode(self, input_ids: mx.array) -> mx.array:
        """Extract embeddings from text chunks"""
        batch_size, seq_len = input_ids.shape
        
        # Accumulator for multi-timestep outputs
        output_accumulator = None
        
        for t in range(self.timesteps):
            # Embedding lookup + positional encoding
            x_embed = self.embedding(input_ids)
            x_embed = x_embed + self.pos_encoding[:seq_len, :]
            
            # Convert to spikes
            embed_spikes, embed_membrane = self.spike_layers['embedding'](x_embed, t)
            
            # Process through transformer layers
            layer_membrane = embed_membrane
            for i, layer in enumerate(self.transformer_layers):
                layer_membrane = self.spike_transformer_block(
                    layer_membrane, layer, i, t
                )
            
            # Accumulate outputs
            if output_accumulator is None:
                output_accumulator = layer_membrane
            else:
                output_accumulator = output_accumulator + layer_membrane
        
        # Average over timesteps
        final_output = output_accumulator / self.timesteps
        
        # Mean pooling over sequence (ignoring padding)
        pad_token_id = self.token_to_idx['<PAD>']
        embeddings = []
        
        for b in range(batch_size):
            # Create mask for non-padding tokens
            mask = (input_ids[b] != pad_token_id).astype(mx.float32)
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

    def get_text_embedding(self, text: str) -> mx.array:
        """Get embedding for a single text"""
        # Tokenize
        tokens = [self.token_to_idx['<START>']]
        tokens += self.preprocess_text(text[:self.max_seq_len-2])
        tokens += [self.token_to_idx['<END>']]
        
        # Pad
        while len(tokens) < self.max_seq_len:
            tokens.append(self.token_to_idx['<PAD>'])
        tokens = tokens[:self.max_seq_len]
        
        # Get embedding
        input_ids = mx.array([tokens])
        embedding = self.encode(input_ids)
        
        return embedding[0]
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> str:
        """Generate text given a prompt using the trained model"""
        # Tokenize prompt
        tokens = self.tokenizer.tokenize(prompt)
        tokens = [self.token_to_idx['<START>']] + tokens
        
        for _ in range(max_length):
            # Prepare input (pad if necessary)
            input_tokens = tokens[-self.max_seq_len:]
            if len(input_tokens) < self.max_seq_len:
                input_tokens = input_tokens + [self.token_to_idx['<PAD>']] * (self.max_seq_len - len(input_tokens))
            
            input_ids = mx.array([input_tokens])
            
            # Get logits
            logits = self(input_ids)
            
            # Get prediction for the last valid position
            last_pos = min(len(tokens) - 1, self.max_seq_len - 1)
            next_logits = logits[0, last_pos, :] / temperature
            
            # Sample next token
            probs = mx.softmax(next_logits)
            next_token = mx.random.categorical(mx.log(probs))
            
            # Stop if END token
            if next_token.item() == self.token_to_idx['<END>']:
                break
            
            # Add to sequence
            tokens.append(next_token.item())
        
        # Decode using tokenizer (excluding START token)
        generated_tokens = [t for t in tokens[1:] if t != self.token_to_idx['<PAD>']]
        return self.tokenizer.decode(generated_tokens)

    def preprocess_text(self, text: str) -> List[int]:
        """Convert text to token indices using the syllable tokenizer"""
        return self.tokenizer.tokenize(text)
    
    def create_training_batch(self, texts: List[str], batch_size: int = 8):
        """Create batches of training data using the tokenizer"""
        batch_x = []
        batch_y = []
        
        for text in texts[:batch_size]:
            # Tokenize with START and END tokens
            tokens = [self.token_to_idx['<START>']]
            tokens += self.preprocess_text(text[:self.max_seq_len-2])
            tokens += [self.token_to_idx['<END>']]
            
            # Pad to max_seq_len
            while len(tokens) < self.max_seq_len:
                tokens.append(self.token_to_idx['<PAD>'])
            
            # Truncate if necessary
            tokens = tokens[:self.max_seq_len]
            
            # Create input (all but last) and target (all but first)
            batch_x.append(tokens[:-1])
            batch_y.append(tokens[1:])
        
        return mx.array(batch_x), mx.array(batch_y)
    
    def compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from loss"""
        return mx.exp(mx.minimum(loss, 100.0))  # Cap to prevent overflow
    
    def train_on_wikipedia(self, data_loader, 
                          epochs: int = 10, 
                          batch_size: int = 8,
                          chunks_per_epoch: int = 1000,
                          checkpoint_dir: str = './checkpoints',
                          save_interval: int = 20):
        """Train on Wikipedia dataset with MLX optimizer"""
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize MLX optimizer
        optimizer = optim.AdamW(
            learning_rate=self.learning_rate,
            betas=[ADAM_BETA1, ADAM_BETA2],
            eps=ADAM_EPSILON,
            weight_decay=0.01
        )
        
        # Create loss and grad function
        def loss_fn(model, batch_x, batch_y):
            _, loss = model(batch_x, batch_y)
            return loss
        
        # Create value and grad function
        loss_and_grad_fn = nn.value_and_grad(self, loss_fn)
        
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
                batch_x, batch_y = self.create_training_batch(batch_texts, len(batch_texts))
                
                # Forward and backward pass
                loss, grads = loss_and_grad_fn(self, batch_x, batch_y)
                
                # Clip gradients
                grads = tree_map(lambda x: mx.clip(x, -GRADIENT_CLIP_VALUE, GRADIENT_CLIP_VALUE), grads)
                
                # Update weights
                optimizer.update(self, grads)
                
                # Evaluate to ensure computation
                mx.eval(self.parameters())
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                n_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'perplexity': f'{self.compute_perplexity(loss).item():.2f}'
                })
            
            # Epoch statistics
            avg_epoch_loss = epoch_loss / n_batches
            epoch_perplexity = self.compute_perplexity(mx.array(avg_epoch_loss)).item()
            
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['loss'].append(avg_epoch_loss)
            self.training_history['perplexity'].append(epoch_perplexity)
            
            print(f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Perplexity: {epoch_perplexity:.2f}")
            
            # Generate sample
            if (epoch + 1) % LOG_INTERVAL == 0:
                prompts = ["The", "What", "A", "I"]
                for prompt in prompts:
                    generated = self.generate(prompt, max_length=100, temperature=0.8)
                    print(f"Sample '{prompt}': {generated[:200]}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(checkpoint_dir, epoch + 1)
    
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
        
        checkpoint = {
            'epoch': epoch,
            'model_params': params,
            'tokenizer': self.tokenizer,
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
        
        if 'tokenizer' in checkpoint:
            self.tokenizer = checkpoint['tokenizer']
            self.vocab = self.tokenizer.vocab
            self.vocab_size = len(self.vocab)
            self.token_to_idx = self.tokenizer.vocab_to_id
            self.idx_to_token = self.tokenizer.id_to_vocab
        
        # Convert numpy arrays back to MLX arrays
        def to_mlx(x):
            if isinstance(x, np.ndarray):
                return mx.array(x)
            return x
        
        # Load parameters
        params = tree_map(to_mlx, checkpoint['model_params'])
        
        # Update model parameters
        self.update(params)
        
        # Restore training history
        self.training_history = checkpoint['training_history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

# WikipediaDataLoader remains the same
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
            chunk_size=2048,
            overlap=64,
            tokenizer=tokenizer
        )
        data_loader.load_articles()
        data_loader.create_chunks()
        model = SpikeLLM(tokenizer=tokenizer)
        print(f"Tokenizer vocab size: {len(model.tokenizer.vocab)}")
        print(f"Tokenizer max ID: {max(model.tokenizer.id_to_vocab.keys())}")
        print(f"Embedding weight shape: {model.embedding.weight.shape}")  # Fixed line
        model.train_on_wikipedia(
            data_loader=data_loader,
            epochs=EPOCHS,
            batch_size=32,
            chunks_per_epoch=5000,
            checkpoint_dir=SAVE_FILEPATH,
            save_interval=1
        )
        # Test embeddings
        embedding1 = model.get_text_embedding("The capital of France")
        embedding2 = model.get_text_embedding("Paris is a city") 
        similarity = mx.sum(embedding1 * embedding2).item()  # MLX dot product
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