import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import os
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm

# Import MLX for model operations
import mlx.core as mx
import mlx.nn as nn

class SpikeLLMAnalyzer:
    """Comprehensive analysis tool for SpikeLLM checkpoints"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        self.model_params = None
        self.config = None
        self.training_history = None
        self.tokenizer = None
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Analysis results storage
        self.weight_stats = {}
        self.activation_stats = defaultdict(list)
        self.spike_stats = defaultdict(list)
        self.generation_trace = []
        
    def _load_checkpoint(self):
        """Load the checkpoint file"""
        print(f"Loading checkpoint from {self.checkpoint_path}")
        with open(self.checkpoint_path, 'rb') as f:
            self.checkpoint = pickle.load(f)
        
        self.model_params = self.checkpoint['model_params']
        self.config = self.checkpoint['config']
        self.training_history = self.checkpoint['training_history']
        self.tokenizer = self.checkpoint.get('tokenizer')
        
        print(f"Loaded checkpoint from epoch {self.checkpoint['epoch']}")
        print(f"Model configuration: {self.config}")
        
    def analyze_training_progress(self, save_path: str = None):
        """Analyze and visualize training history"""
        if not self.training_history or not self.training_history['epoch']:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot loss
        epochs = self.training_history['epoch']
        losses = self.training_history['loss']
        ax1.plot(epochs, losses, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot perplexity
        perplexities = self.training_history['perplexity']
        ax2.plot(epochs, perplexities, 'r-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Perplexity Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved training progress plot to {save_path}")
        else:
            plt.show()
        
        # Print statistics
        print("\nTraining Statistics:")
        print(f"Final Loss: {losses[-1]:.4f}")
        print(f"Final Perplexity: {perplexities[-1]:.2f}")
        print(f"Loss Reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
        print(f"Perplexity Reduction: {(perplexities[0] - perplexities[-1]) / perplexities[0] * 100:.1f}%")
        
    def analyze_weight_distribution(self, save_path: str = None):
        """Analyze weight distributions across layers"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        layer_types = ['embedding', 'attention', 'feedforward', 'output', 'layernorm', 'all']
        
        for idx, layer_type in enumerate(layer_types):
            ax = axes[idx]
            weights = []
            
            for param_name, param_value in self.model_params.items():
                if layer_type == 'all' or self._classify_parameter(param_name) == layer_type:
                    if isinstance(param_value, np.ndarray):
                        weights.extend(param_value.flatten())
            
            if weights:
                weights = np.array(weights)
                
                # Remove extreme outliers for better visualization
                q1, q99 = np.percentile(weights, [1, 99])
                weights_clipped = weights[(weights >= q1) & (weights <= q99)]
                
                ax.hist(weights_clipped, bins=100, alpha=0.7, color='blue', edgecolor='black')
                ax.set_title(f'{layer_type.capitalize()} Weights Distribution')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Frequency')
                
                # Add statistics
                stats_text = f'μ={np.mean(weights):.3f}\nσ={np.std(weights):.3f}\nmin={np.min(weights):.3f}\nmax={np.max(weights):.3f}'
                ax.text(0.7, 0.7, stats_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Store statistics
                self.weight_stats[layer_type] = {
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights)),
                    'min': float(np.min(weights)),
                    'max': float(np.max(weights)),
                    'sparsity': float(np.sum(np.abs(weights) < 0.01) / len(weights))
                }
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved weight distribution plot to {save_path}")
        else:
            plt.show()
        
        # Print weight statistics
        print("\nWeight Statistics Summary:")
        print("-" * 60)
        print(f"{'Layer Type':<15} {'Mean':>10} {'Std':>10} {'Sparsity':>10}")
        print("-" * 60)
        for layer_type, stats in self.weight_stats.items():
            print(f"{layer_type:<15} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['sparsity']:>10.2%}")
    
    def _classify_parameter(self, param_name: str) -> str:
        """Classify parameter by type"""
        if 'embedding' in param_name:
            return 'embedding'
        elif any(x in param_name for x in ['w_q', 'w_k', 'w_v', 'w_o']):
            return 'attention'
        elif 'ff' in param_name:
            return 'feedforward'
        elif 'out_proj' in param_name:
            return 'output'
        elif 'ln' in param_name:
            return 'layernorm'
        else:
            return 'other'
    
    def analyze_layer_importance(self, save_path: str = None):
        """Analyze which layers have the most significant weights"""
        layer_importance = {}
        
        # Group parameters by layer
        for layer_idx in range(self.config['n_layers']):
            layer_weights = []
            layer_params = []
            
            for param_name, param_value in self.model_params.items():
                if f'layer_{layer_idx}' in param_name:
                    if isinstance(param_value, np.ndarray):
                        layer_weights.extend(np.abs(param_value.flatten()))
                        layer_params.append(param_name)
            
            if layer_weights:
                layer_importance[f'layer_{layer_idx}'] = {
                    'mean_magnitude': np.mean(layer_weights),
                    'max_magnitude': np.max(layer_weights),
                    'weight_norm': np.linalg.norm(layer_weights),
                    'n_params': len(layer_weights),
                    'param_names': layer_params
                }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        layers = sorted(layer_importance.keys(), key=lambda x: int(x.split('_')[1]))
        mean_mags = [layer_importance[l]['mean_magnitude'] for l in layers]
        weight_norms = [layer_importance[l]['weight_norm'] for l in layers]
        
        x = range(len(layers))
        ax1.bar(x, mean_mags, alpha=0.7, color='blue')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Mean Weight Magnitude')
        ax1.set_title('Average Weight Magnitude by Layer')
        ax1.set_xticks(x)
        ax1.set_xticklabels([l.split('_')[1] for l in layers])
        
        ax2.bar(x, weight_norms, alpha=0.7, color='green')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Weight Norm')
        ax2.set_title('Total Weight Norm by Layer')
        ax2.set_xticks(x)
        ax2.set_xticklabels([l.split('_')[1] for l in layers])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved layer importance plot to {save_path}")
        else:
            plt.show()
        
        # Identify most and least important layers
        sorted_layers = sorted(layer_importance.items(), 
                             key=lambda x: x[1]['mean_magnitude'], 
                             reverse=True)
        
        print("\nLayer Importance Ranking (by mean weight magnitude):")
        print("-" * 50)
        for i, (layer, stats) in enumerate(sorted_layers[:5]):
            print(f"{i+1}. {layer}: {stats['mean_magnitude']:.6f}")
        
        return layer_importance
    
    def trace_generation(self, model, prompt: str, max_length: int = 50):
        """Trace spike activity during text generation"""
        print(f"\nTracing generation for prompt: '{prompt}'")
        
        # This would require the actual model instance
        # For now, we'll analyze the static weights
        print("Note: Dynamic spike tracing requires the model instance")
        
        # Analyze prompt tokens
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(prompt)
            print(f"Prompt tokens: {tokens}")
            print(f"Token count: {len(tokens)}")
            
            # Analyze token frequency in vocabulary
            token_strings = [self.tokenizer.id_to_vocab.get(t, '<UNK>') for t in tokens]
            print(f"Token strings: {token_strings}")
    
    def analyze_embedding_space(self, save_path: str = None):
        """Analyze the learned embedding space"""
        if 'embedding.weight' not in self.model_params:
            print("No embedding weights found")
            return
        
        embeddings = self.model_params['embedding.weight']
        print(f"Embedding shape: {embeddings.shape}")
        
        # Compute embedding statistics
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot embedding norm distribution
        ax1.hist(embedding_norms, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax1.set_xlabel('Embedding Norm')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Embedding Norms')
        ax1.axvline(np.mean(embedding_norms), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(embedding_norms):.3f}')
        ax1.legend()
        
        # Plot top tokens by embedding norm
        if self.tokenizer:
            top_indices = np.argsort(embedding_norms)[-20:][::-1]
            top_tokens = [self.tokenizer.id_to_vocab.get(idx, f'<ID:{idx}>') for idx in top_indices]
            top_norms = embedding_norms[top_indices]
            
            y_pos = np.arange(len(top_tokens))
            ax2.barh(y_pos, top_norms, alpha=0.7, color='orange')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_tokens)
            ax2.set_xlabel('Embedding Norm')
            ax2.set_title('Top 20 Tokens by Embedding Norm')
            ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved embedding analysis plot to {save_path}")
        else:
            plt.show()
        
        # Analyze special tokens
        if self.tokenizer:
            special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
            print("\nSpecial Token Embeddings:")
            print("-" * 40)
            for token in special_tokens:
                if token in self.tokenizer.vocab_to_id:
                    idx = self.tokenizer.vocab_to_id[token]
                    norm = embedding_norms[idx]
                    print(f"{token}: norm={norm:.4f}")
    
    def analyze_attention_patterns(self, save_path: str = None):
        """Analyze attention weight patterns"""
        attention_stats = {}
        
        for layer_idx in range(self.config['n_layers']):
            layer_stats = {}
            
            # Analyze Q, K, V, O matrices
            for matrix_type in ['w_q', 'w_k', 'w_v', 'w_o']:
                param_name = f'layer_{layer_idx}.{matrix_type}.weight'
                if param_name in self.model_params:
                    weights = self.model_params[param_name]
                    
                    # Compute singular values to understand information flow
                    try:
                        U, s, Vt = np.linalg.svd(weights, full_matrices=False)
                        layer_stats[matrix_type] = {
                            'top_singular_values': s[:10].tolist(),
                            'effective_rank': np.sum(s > 0.1),
                            'condition_number': s[0] / s[-1] if s[-1] > 0 else np.inf
                        }
                    except:
                        layer_stats[matrix_type] = {'error': 'SVD failed'}
            
            attention_stats[f'layer_{layer_idx}'] = layer_stats
        
        # Visualize singular value decay
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, matrix_type in enumerate(['w_q', 'w_k', 'w_v', 'w_o']):
            ax = axes[idx]
            
            for layer_idx in range(min(4, self.config['n_layers'])):  # Show first 4 layers
                stats = attention_stats.get(f'layer_{layer_idx}', {}).get(matrix_type, {})
                if 'top_singular_values' in stats:
                    s_values = stats['top_singular_values']
                    ax.plot(range(len(s_values)), s_values, 
                           label=f'Layer {layer_idx}', marker='o')
            
            ax.set_xlabel('Singular Value Index')
            ax.set_ylabel('Singular Value')
            ax.set_title(f'{matrix_type.upper()} Singular Value Decay')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved attention analysis plot to {save_path}")
        else:
            plt.show()
        
        return attention_stats
    
    def analyze_sparsity(self):
        """Analyze weight sparsity across the model"""
        sparsity_by_layer = {}
        
        for param_name, param_value in self.model_params.items():
            if isinstance(param_value, np.ndarray):
                # Calculate sparsity (percentage of near-zero weights)
                near_zero = np.abs(param_value) < 0.01
                sparsity = np.sum(near_zero) / param_value.size
                
                sparsity_by_layer[param_name] = {
                    'sparsity': float(sparsity),
                    'total_params': param_value.size,
                    'near_zero_params': int(np.sum(near_zero))
                }
        
        # Sort by sparsity
        sorted_layers = sorted(sparsity_by_layer.items(), 
                             key=lambda x: x[1]['sparsity'], 
                             reverse=True)
        
        print("\nWeight Sparsity Analysis:")
        print("-" * 70)
        print(f"{'Parameter':<40} {'Sparsity':>10} {'Near-Zero':>10} {'Total':>10}")
        print("-" * 70)
        
        for param_name, stats in sorted_layers[:15]:  # Show top 15
            print(f"{param_name[:40]:<40} {stats['sparsity']:>10.2%} "
                  f"{stats['near_zero_params']:>10,} {stats['total_params']:>10,}")
        
        # Calculate overall sparsity
        total_params = sum(s['total_params'] for s in sparsity_by_layer.values())
        total_near_zero = sum(s['near_zero_params'] for s in sparsity_by_layer.values())
        overall_sparsity = total_near_zero / total_params
        
        print("-" * 70)
        print(f"{'OVERALL':<40} {overall_sparsity:>10.2%} "
              f"{total_near_zero:>10,} {total_params:>10,}")
        
        return sparsity_by_layer
    
    def generate_report(self, output_dir: str = 'analysis_output'):
        """Generate a comprehensive analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating comprehensive analysis report...")
        
        # 1. Training progress
        print("\n1. Analyzing training progress...")
        self.analyze_training_progress(os.path.join(output_dir, 'training_progress.png'))
        
        # 2. Weight distributions
        print("\n2. Analyzing weight distributions...")
        self.analyze_weight_distribution(os.path.join(output_dir, 'weight_distributions.png'))
        
        # 3. Layer importance
        print("\n3. Analyzing layer importance...")
        layer_importance = self.analyze_layer_importance(os.path.join(output_dir, 'layer_importance.png'))
        
        # 4. Embedding space
        print("\n4. Analyzing embedding space...")
        self.analyze_embedding_space(os.path.join(output_dir, 'embedding_analysis.png'))
        
        # 5. Attention patterns
        print("\n5. Analyzing attention patterns...")
        attention_stats = self.analyze_attention_patterns(os.path.join(output_dir, 'attention_analysis.png'))
        
        # 6. Sparsity analysis
        print("\n6. Analyzing weight sparsity...")
        sparsity_stats = self.analyze_sparsity()
        
        # Generate summary report
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write("SpikeLLM Model Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Epoch: {self.checkpoint['epoch']}\n")
            f.write(f"Model Configuration:\n")
            for key, value in self.config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            if self.training_history and self.training_history['epoch']:
                f.write("Training Summary:\n")
                f.write(f"  Final Loss: {self.training_history['loss'][-1]:.4f}\n")
                f.write(f"  Final Perplexity: {self.training_history['perplexity'][-1]:.2f}\n")
                f.write("\n")
            
            f.write("Weight Statistics:\n")
            for layer_type, stats in self.weight_stats.items():
                f.write(f"  {layer_type}:\n")
                for key, value in stats.items():
                    f.write(f"    {key}: {value:.4f}\n")
            f.write("\n")
            
            # Add vocabulary analysis if tokenizer available
            if self.tokenizer:
                f.write("Tokenizer Statistics:\n")
                f.write(f"  Vocabulary size: {len(self.tokenizer.vocab)}\n")
                f.write(f"  Special tokens: {[t for t in self.tokenizer.vocab if '<' in t and '>' in t]}\n")
                f.write("\n")
        
        print(f"\nAnalysis complete! Results saved to {output_dir}/")
        print(f"Report saved to {report_path}")


def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze SpikeLLM checkpoint')
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint .pkl file')
    parser.add_argument('--output_dir', type=str, default='analysis_output', 
                       help='Directory to save analysis results')
    parser.add_argument('--trace_prompt', type=str, default=None,
                       help='Prompt to trace during generation')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SpikeLLMAnalyzer(args.checkpoint_path)
    
    # Generate full report
    analyzer.generate_report(args.output_dir)
    
    # Optional: trace generation
    if args.trace_prompt:
        print(f"\nTracing generation for: '{args.trace_prompt}'")
        # This would require loading the actual model
        analyzer.trace_generation(None, args.trace_prompt)


if __name__ == "__main__":
    # Example usage without command line arguments
    checkpoint_path = "checkpointsMLX/spike_llm_mlx_epoch_260.pkl"  # Update this path
    
    if os.path.exists(checkpoint_path):
        analyzer = SpikeLLMAnalyzer(checkpoint_path)
        analyzer.generate_report('analysis_output')
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please update the checkpoint_path variable or use command line arguments:")
        print("python analyze_spike_llm.py path/to/checkpoint.pkl")