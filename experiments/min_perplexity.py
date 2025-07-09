import os
import json
import pickle
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from tokenizer2 import SyllableTokenizer

class EntropyCalculator:
    def __init__(self, tokenizer_path='syllable_tokenizer.pkl'):
        """Initialize with the syllable tokenizer"""
        print(f"Loading tokenizer from {tokenizer_path}...")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"Tokenizer loaded. Vocabulary size: {len(self.tokenizer.vocab)}")
        
        self.token_counts = Counter()
        self.bigram_counts = Counter()
        self.total_tokens = 0
        
    def process_wikipedia_files(self, folder_path: str, max_articles: int = None):
        """Process Wikipedia JSON files to count token frequencies"""
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files to process")
        
        articles_processed = 0
        
        for json_file in tqdm(json_files, desc="Processing files"):
            if max_articles and articles_processed >= max_articles:
                break
                
            file_path = os.path.join(folder_path, json_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                
                for article in articles:
                    if max_articles and articles_processed >= max_articles:
                        break
                        
                    if 'content' in article and article['content']:
                        content = article['content'][:20000]  # Limit per article
                        
                        # Tokenize the content
                        tokens = self.tokenizer.tokenize(content)
                        
                        # Update counts
                        self.token_counts.update(tokens)
                        self.total_tokens += len(tokens)
                        
                        # Count bigrams for conditional entropy
                        for i in range(len(tokens) - 1):
                            bigram = (tokens[i], tokens[i+1])
                            self.bigram_counts[bigram] += 1
                        
                        articles_processed += 1
                        
            except Exception as e:
                print(f"\nError processing {json_file}: {e}")
                continue
        
        print(f"\nProcessed {articles_processed} articles")
        print(f"Total tokens: {self.total_tokens:,}")
        print(f"Unique tokens used: {len(self.token_counts):,} out of {len(self.tokenizer.vocab)} in vocabulary")
        
    def calculate_entropy(self) -> Tuple[float, float]:
        """Calculate entropy and perplexity from token counts"""
        if self.total_tokens == 0:
            raise ValueError("No tokens counted. Run process_wikipedia_files first.")
        
        # Calculate unigram entropy
        entropy = 0.0
        for token, count in self.token_counts.items():
            p = count / self.total_tokens
            if p > 0:
                entropy -= p * np.log2(p)
        
        perplexity = 2 ** entropy
        
        return entropy, perplexity
    
    def calculate_conditional_entropy(self) -> Tuple[float, float]:
        """Calculate conditional entropy H(X|Y) using bigrams"""
        if not self.bigram_counts:
            return None, None
        
        conditional_entropy = 0.0
        
        # Group bigrams by first token
        first_token_counts = Counter()
        for (t1, t2), count in self.bigram_counts.items():
            first_token_counts[t1] += count
        
        # Calculate H(X|Y) = sum over y of p(y) * H(X|Y=y)
        for first_token, total_count in first_token_counts.items():
            p_first = total_count / sum(self.bigram_counts.values())
            
            # Calculate H(X|Y=first_token)
            entropy_given_first = 0.0
            for (t1, t2), count in self.bigram_counts.items():
                if t1 == first_token:
                    p_second_given_first = count / total_count
                    if p_second_given_first > 0:
                        entropy_given_first -= p_second_given_first * np.log2(p_second_given_first)
            
            conditional_entropy += p_first * entropy_given_first
        
        conditional_perplexity = 2 ** conditional_entropy
        
        return conditional_entropy, conditional_perplexity
    
    def analyze_token_distribution(self, top_n: int = 50):
        """Analyze and visualize token distribution"""
        print(f"\nTop {top_n} most frequent tokens:")
        print("-" * 60)
        
        cumulative_prob = 0.0
        for i, (token_id, count) in enumerate(self.token_counts.most_common(top_n)):
            token = self.tokenizer.id_to_vocab[token_id]
            prob = count / self.total_tokens
            cumulative_prob += prob
            
            # Format token for display
            display_token = token
            if token == '\n':
                display_token = '\\n'
            elif token == '\t':
                display_token = '\\t'
            elif token == ' ':
                display_token = '<SPACE>'
                
            print(f"{i+1:3d}. '{display_token:15s}' : {count:8,d} ({prob*100:5.2f}%) [cum: {cumulative_prob*100:5.2f}%]")
        
        # Coverage analysis
        print(f"\nToken coverage:")
        for coverage in [0.5, 0.8, 0.9, 0.95, 0.99]:
            tokens_needed = self._tokens_for_coverage(coverage)
            print(f"  {coverage*100:.0f}% coverage: {tokens_needed} tokens ({tokens_needed/len(self.tokenizer.vocab)*100:.1f}% of vocabulary)")
    
    def _tokens_for_coverage(self, coverage: float) -> int:
        """Calculate how many tokens are needed for X% coverage"""
        cumulative = 0.0
        for i, (_, count) in enumerate(self.token_counts.most_common()):
            cumulative += count / self.total_tokens
            if cumulative >= coverage:
                return i + 1
        return len(self.token_counts)
    
    def plot_distributions(self):
        """Create visualization plots"""
        # Prepare data
        frequencies = [count for _, count in self.token_counts.most_common()]
        ranks = list(range(1, len(frequencies) + 1))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Zipf plot (log-log)
        ax = axes[0, 0]
        ax.loglog(ranks[:1000], frequencies[:1000], 'b-', alpha=0.7)
        ax.set_xlabel('Token Rank')
        ax.set_ylabel('Frequency')
        ax.set_title('Token Frequency Distribution (Zipf\'s Law)')
        ax.grid(True, alpha=0.3)
        
        # 2. Cumulative probability
        ax = axes[0, 1]
        cumulative_probs = np.cumsum(frequencies) / self.total_tokens
        ax.plot(ranks, cumulative_probs, 'g-')
        ax.set_xlabel('Number of Tokens')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Token Coverage')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1000)
        
        # 3. Token length distribution
        ax = axes[1, 0]
        token_lengths = Counter()
        for token_id, count in self.token_counts.items():
            token = self.tokenizer.id_to_vocab[token_id]
            if not token.startswith('<'):  # Skip special tokens
                token_lengths[len(token)] += count
        
        lengths = sorted(token_lengths.keys())
        counts = [token_lengths[l] for l in lengths]
        ax.bar(lengths, counts, alpha=0.7)
        ax.set_xlabel('Token Length (characters)')
        ax.set_ylabel('Frequency')
        ax.set_title('Token Length Distribution')
        ax.set_xlim(0, 15)
        
        # 4. Entropy contribution by rank
        ax = axes[1, 1]
        entropy_contributions = []
        for _, count in self.token_counts.most_common(500):
            p = count / self.total_tokens
            contribution = -p * np.log2(p) if p > 0 else 0
            entropy_contributions.append(contribution)
        
        ax.plot(range(1, len(entropy_contributions) + 1), entropy_contributions, 'r-', alpha=0.7)
        ax.set_xlabel('Token Rank')
        ax.set_ylabel('Entropy Contribution (bits)')
        ax.set_title('Entropy Contribution by Token Rank')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('token_distribution_analysis.png', dpi=150)
        plt.show()
    
    def calculate_theoretical_limits(self):
        """Calculate various theoretical limits for the vocabulary"""
        vocab_size = len(self.tokenizer.vocab)
        active_vocab_size = len(self.token_counts)
        
        print("\n" + "="*60)
        print("THEORETICAL LIMITS AND ACTUAL MEASUREMENTS")
        print("="*60)
        
        # Theoretical limits
        print("\nTheoretical Limits:")
        print(f"  Vocabulary size: {vocab_size:,}")
        print(f"  Maximum entropy (uniform): {np.log2(vocab_size):.4f} bits")
        print(f"  Maximum perplexity (uniform): {vocab_size:,}")
        print(f"  Minimum entropy: 0 bits (perfect prediction)")
        print(f"  Minimum perplexity: 1")
        
        # Actual measurements
        entropy, perplexity = self.calculate_entropy()
        print("\nActual Measurements (Unigram):")
        print(f"  Active vocabulary: {active_vocab_size:,} tokens ({active_vocab_size/vocab_size*100:.1f}%)")
        print(f"  Entropy: {entropy:.4f} bits")
        print(f"  Perplexity: {perplexity:.2f}")
        
        # Conditional entropy (bigram)
        cond_entropy, cond_perplexity = self.calculate_conditional_entropy()
        if cond_entropy:
            print("\nConditional Entropy (Bigram):")
            print(f"  H(X|Y): {cond_entropy:.4f} bits")
            print(f"  Perplexity: {cond_perplexity:.2f}")
            print(f"  Reduction from unigram: {entropy - cond_entropy:.4f} bits ({(entropy - cond_entropy)/entropy*100:.1f}%)")
        
        # Estimate minimum achievable perplexity
        print("\nEstimated Minimum Achievable Perplexity:")
        
        # Based on token type distribution
        special_tokens = sum(1 for t in self.token_counts if self.tokenizer.id_to_vocab[t].startswith('<'))
        single_chars = sum(1 for t in self.token_counts if len(self.tokenizer.id_to_vocab[t]) == 1 and not self.tokenizer.id_to_vocab[t].startswith('<'))
        syllables = active_vocab_size - special_tokens - single_chars
        
        print(f"\nToken Type Distribution:")
        print(f"  Special tokens: {special_tokens}")
        print(f"  Single characters: {single_chars}")
        print(f"  Syllables/words: {syllables}")
        
        # Estimate based on token types
        # Single chars are hard to predict (high entropy)
        # Syllables/words are easier (lower entropy)
        char_entropy_estimate = 4.5  # bits, for single characters
        syllable_entropy_estimate = 3.0  # bits, for syllables/words
        
        # Weight by usage
        char_usage = sum(self.token_counts[t] for t in self.token_counts if len(self.tokenizer.id_to_vocab[t]) == 1 and not self.tokenizer.id_to_vocab[t].startswith('<'))
        char_usage_ratio = char_usage / self.total_tokens
        
        estimated_min_entropy = char_usage_ratio * char_entropy_estimate + (1 - char_usage_ratio) * syllable_entropy_estimate
        estimated_min_perplexity = 2 ** estimated_min_entropy
        
        print(f"\nEstimated minimum (with perfect model):")
        print(f"  Character usage: {char_usage_ratio*100:.1f}% of tokens")
        print(f"  Estimated minimum entropy: {estimated_min_entropy:.2f} bits")
        print(f"  Estimated minimum perplexity: {estimated_min_perplexity:.1f}")
        
        print(f"\nYour model's current performance:")
        print(f"  Current perplexity: 18.7")
        print(f"  Distance from theoretical minimum: {18.7 / estimated_min_perplexity:.2f}x")
        print(f"  Potential improvement: {(18.7 - estimated_min_perplexity) / 18.7 * 100:.1f}%")


def main():
    # Initialize calculator
    calc = EntropyCalculator('syllable_tokenizer.pkl')
    
    # Process Wikipedia data
    folder_path = 'scraped_data'
    calc.process_wikipedia_files(folder_path, max_articles=3000)  # Adjust number as needed
    
    # Calculate and display results
    calc.calculate_theoretical_limits()
    calc.analyze_token_distribution(top_n=50)
    
    # Create visualizations
    calc.plot_distributions()
    
    # Save results
    results = {
        'total_tokens': calc.total_tokens,
        'unique_tokens': len(calc.token_counts),
        'vocab_size': len(calc.tokenizer.vocab),
        'entropy': calc.calculate_entropy()[0],
        'perplexity': calc.calculate_entropy()[1],
        'token_counts': dict(calc.token_counts.most_common(1000))  # Save top 1000
    }
    
    with open('entropy_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to entropy_analysis_results.json")


if __name__ == "__main__":
    main()