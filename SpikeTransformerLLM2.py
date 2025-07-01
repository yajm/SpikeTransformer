import numpy as np
import pickle
import json
from typing import Dict, List, Tuple
import random
import os
from tqdm import tqdm

# Add these imports to your existing code
import gc  # For garbage collection
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.info(f"Loading Wikipedia articles from {self.json_path}")
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
        
        logging.info(f"Loaded {len(self.articles)} articles")
        
    def create_chunks(self):
        """Split articles into overlapping chunks for training"""
        logging.info("Creating training chunks...")
        
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
        
        logging.info(f"Created {len(self.chunks)} training chunks")
        
    def get_training_chunks(self, num_chunks: int = None):
        """Get training chunks, optionally limiting the number"""
        if num_chunks:
            return random.sample(self.chunks, min(num_chunks, len(self.chunks)))
        return self.chunks

# Modify the SpikeLLM class to add these methods:

class SpikeLLMEnhanced(SpikeLLM):
    """Enhanced SpikeLLM with better training capabilities for large datasets"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_history = {
            'epoch': [],
            'loss': [],
            'perplexity': []
        }
        
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
        
        logging.info(f"Starting training on Wikipedia dataset")
        logging.info(f"Epochs: {epochs}, Batch size: {batch_size}, Chunks per epoch: {chunks_per_epoch}")
        
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
                for j in range(len(batch_x)):
                    x = batch_x[j:j+1]  # Single example
                    y = batch_y[j]
                    
                    # Forward pass
                    logits = self.forward(x)
                    
                    # Compute loss
                    loss = 0
                    for k in range(len(y)):
                        if y[k] != self.char_to_idx['<PAD>']:
                            probs = self.softmax(logits[0, k, :])
                            loss -= np.log(probs[y[k]] + 1e-10)
                    
                    batch_loss += loss
                    
                    # Backward pass (simplified)
                    self.simple_backward(x, y, logits)
                
                # Update metrics
                avg_batch_loss = batch_loss / len(batch_x)
                epoch_loss += avg_batch_loss
                n_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{avg_batch_loss:.4f}',
                    'perplexity': f'{self.compute_perplexity(avg_batch_loss):.2f}'
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
            
            logging.info(f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Perplexity: {epoch_perplexity:.2f}")
            
            # Generate sample
            if (epoch + 1) % LOG_INTERVAL == 0:
                prompts = ["The ", "In ", "A ", "Wikipedia "]
                for prompt in prompts:
                    generated = self.generate(prompt, max_length=100, temperature=0.8)
                    logging.info(f"Sample '{prompt}': {generated[:200]}...")
            
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
        
        logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        
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
        
        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

# Main training script
if __name__ == "__main__":
    # Configuration
    WIKIPEDIA_JSON_PATH = "wikipedia_top_articles.json"
    
    # Model hyperparameters (adjusted for larger dataset)
    config = {
        'max_seq_len': 512,      # Increased sequence length
        'd_model': 256,          # Increased model dimension
        'n_heads': 8,
        'n_layers': 6,           # More layers for complex patterns
        'd_ff': 1024,            # Larger feedforward dimension
        'timesteps': 4,
        'learning_rate': 0.0005  # Lower learning rate for stability
    }
    
    # Training parameters
    EPOCHS = 50
    BATCH_SIZE = 4  # Smaller batch size due to longer sequences
    CHUNKS_PER_EPOCH = 2000  # Number of chunks to use per epoch
    CHECKPOINT_INTERVAL = 5
    
    # Initialize data loader
    data_loader = WikipediaDataLoader(
        json_path=WIKIPEDIA_JSON_PATH,
        max_article_length=50000,  # Use first 50k chars of each article
        chunk_size=512,            # Match model's max_seq_len
        overlap=64                 # Overlap between chunks
    )
    
    # Load and preprocess data
    data_loader.load_articles()
    data_loader.create_chunks()
    
    # Create enhanced model
    model = SpikeLLMEnhanced(**config)
    
    # Optional: Load from checkpoint if resuming training
    # model.load_checkpoint('./checkpoints/spike_llm_epoch_10.pkl')
    
    # Train on Wikipedia
    model.train_on_wikipedia(
        data_loader=data_loader,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        chunks_per_epoch=CHUNKS_PER_EPOCH,
        checkpoint_dir='./checkpoints',
        save_interval=CHECKPOINT_INTERVAL
    )
    
    # Save final model
    model.save_checkpoint('./checkpoints', epoch=EPOCHS)
    
    # Test generation
    print("\n" + "="*60)
    print("Final Text Generation Examples:")
    print("="*60)
    
    test_prompts = [
        "The history of ",
        "In 2025, ",
        "Wikipedia is ",
        "The most important ",
        "Scientists discovered "
    ]
    
    for prompt in test_prompts:
        generated = model.generate(prompt, max_length=200, temperature=0.8)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")