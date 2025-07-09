import json
from collections import Counter
import pickle
import re
import heapq
from typing import List, Tuple, Dict, Set

class SmartNgramTokenizer:
    def __init__(self, base_vocab):
        """Initialize with base vocabulary."""
        self.base_vocab = base_vocab
        self.vocab = base_vocab.copy()
        self.vocab_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_vocab = {idx: token for idx, token in enumerate(self.vocab)}
        self.ngram_set = set()  # Track selected n-grams
        
    def calculate_compression_value(self, ngram: str, frequency: int) -> float:
        """
        Calculate the compression value of an n-gram.
        Compression value = (tokens_saved) * frequency
        """
        # Tokens saved = length of n-gram - 1 (since it becomes 1 token)
        tokens_saved = len(ngram) - 1
        return tokens_saved * frequency
    
    def has_overlap(self, ngram: str, selected_ngrams: Set[str]) -> bool:
        """
        Check if an n-gram has significant overlap with already selected n-grams.
        We avoid selecting n-grams that are substrings of already selected ones
        or that contain already selected ones (unless they provide significant value).
        """
        # Check if this n-gram is a substring of any selected n-gram
        for selected in selected_ngrams:
            if ngram in selected and len(selected) > len(ngram):
                return True
        
        # Check if any selected n-gram is a substring of this n-gram
        # (we might still want to add it if it provides enough value)
        for selected in selected_ngrams:
            if selected in ngram and len(selected) < len(ngram):
                # Allow if the new n-gram is significantly longer and useful
                if len(ngram) - len(selected) < 2:
                    return True
        
        return False
    
    def is_valid_ngram(self, ngram: str) -> bool:
        """
        Check if an n-gram is valid according to our rules:
        - Can have at most one space, and only at the beginning
        - Must contain at least one non-space character
        """
        if not ngram:
            return False
        
        # Must contain at least one non-space character
        if ngram.strip() == '':
            return False
        
        # Count spaces in the ngram
        space_count = ngram.count(' ')
        
        # No spaces is valid
        if space_count == 0:
            return True
        
        # Exactly one space is only valid if it's at the beginning
        if space_count == 1 and ngram[0] == ' ':
            return True
        
        # Any other case with spaces is invalid
        return False
    
    def select_top_ngrams(self, ngram_counters: Dict[int, Counter], 
                         max_ngrams: int = 10000, 
                         min_frequency: int = 100) -> List[Tuple[str, int, float]]:
        """
        Select top n-grams based on compression value, avoiding redundancy.
        
        Returns list of (ngram, frequency, compression_value) tuples.
        """
        # Calculate compression values for all n-grams
        candidates = []
        
        for n, counter in ngram_counters.items():
            for ngram, freq in counter.items():
                if freq >= min_frequency and self.is_valid_ngram(ngram):  # Add validity check
                    compression = self.calculate_compression_value(ngram, freq)
                    # Use negative compression for max heap
                    heapq.heappush(candidates, (-compression, ngram, freq, n))
        
        # Select top n-grams greedily
        selected_ngrams = []
        selected_set = set()
        
        while candidates and len(selected_ngrams) < max_ngrams:
            neg_compression, ngram, freq, n = heapq.heappop(candidates)
            compression = -neg_compression
            
            # Skip if it has too much overlap with already selected n-grams
            if not self.has_overlap(ngram, selected_set):
                selected_ngrams.append((ngram, freq, compression))
                selected_set.add(ngram)
        
        return selected_ngrams
    
    def build_vocabulary(self, selected_ngrams: List[Tuple[str, int, float]]):
        """Add selected n-grams to vocabulary."""
        for ngram, freq, compression in selected_ngrams:
            if ngram not in self.vocab_to_id:
                idx = len(self.vocab)
                self.vocab.append(ngram)
                self.vocab_to_id[ngram] = idx
                self.id_to_vocab[idx] = ngram
                self.ngram_set.add(ngram)
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using greedy longest-match approach.
        """
        tokens = []
        i = 0
        
        # Convert to lowercase for matching (but preserve original for CAP markers)
        text_lower = text.lower()
        
        while i < len(text):
            # Check for capitalization
            if i < len(text) and text[i].isupper() and text[i].lower() in self.vocab_to_id:
                tokens.append(self.vocab_to_id['<CAP>'])
            
            # Try to match the longest n-gram first
            matched = False
            for length in range(min(5, len(text) - i), 0, -1):  # Try 5, 4, 3, 2, 1
                substr = text_lower[i:i+length]
                if substr in self.vocab_to_id:
                    tokens.append(self.vocab_to_id[substr])
                    i += length
                    matched = True
                    break
            
            if not matched:
                # Unknown character
                tokens.append(self.vocab_to_id['<UNK>'])
                i += 1
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        result = []
        i = 0
        
        while i < len(token_ids):
            token_id = token_ids[i]
            
            if token_id == self.vocab_to_id['<CAP>'] and i + 1 < len(token_ids):
                # Next token should be capitalized
                i += 1
                next_token = self.id_to_vocab[token_ids[i]]
                result.append(next_token.upper() if len(next_token) == 1 else next_token.capitalize())
            else:
                result.append(self.id_to_vocab.get(token_id, '<UNK>'))
            
            i += 1
        
        return ''.join(result)
    
    def save_vocabulary(self, filename: str):
        """Save vocabulary to file."""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, token in enumerate(self.vocab):
                f.write(f"{i}\t{repr(token)}\n")
    
    def analyze_compression(self, text: str) -> Dict[str, float]:
        """Analyze compression statistics for a given text."""
        tokens = self.tokenize(text)
        original_chars = len(text)
        compressed_tokens = len(tokens)
        
        # Count n-gram usage
        ngram_usage = Counter()
        for token_id in tokens:
            token = self.id_to_vocab[token_id]
            if token in self.ngram_set:
                ngram_usage[token] += 1
        
        return {
            'original_chars': original_chars,
            'compressed_tokens': compressed_tokens,
            'compression_ratio': original_chars / compressed_tokens if compressed_tokens > 0 else 0,
            'savings_percentage': (1 - compressed_tokens / original_chars) * 100 if original_chars > 0 else 0,
            'ngram_usage': ngram_usage.most_common(20)
        }

def process_and_select_ngrams(folder_path: str, base_vocab: List[str], 
                             max_ngrams: int = 10000,
                             min_frequency: int = 100) -> SmartNgramTokenizer:
    """
    Process Wikipedia JSON files from folder and select optimal n-grams for tokenization.
    """
    import os
    from collections import Counter
    
    ngram_counters = {2: Counter(), 3: Counter(), 4: Counter(), 5: Counter(), 6: Counter(), 7: Counter(), 8: Counter(), 9: Counter(), 10: Counter()}
    
    # Get all JSON files from the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} JSON files to process")
    
    total_articles = 0
    for file_idx, json_file in enumerate(json_files):
        file_path = os.path.join(folder_path, json_file)
        print(f"\nProcessing {json_file} ({file_idx + 1}/{len(json_files)})...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            for i, article in enumerate(articles):
                total_articles += 1
                content = article.get('content', '').lower()
                content = re.sub(r'\s+', ' ', content)
                
                for n in [2, 3, 4, 5, 6, 7, 8, 9]:
                    for j in range(len(content) - n + 1):
                        ngram = content[j:j + n]
                        ngram_counters[n][ngram] += 1
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1} articles from {json_file}...")
    
    print(f"\nTotal articles processed: {total_articles}")
    
    # Create tokenizer and select n-grams
    print("\nSelecting optimal n-grams...")
    tokenizer = SmartNgramTokenizer(base_vocab)
    selected_ngrams = tokenizer.select_top_ngrams(ngram_counters, max_ngrams, min_frequency)
    
    print(f"\nSelected {len(selected_ngrams)} n-grams")
    print("\nTop 100 n-grams by compression value:")
    for ngram, freq, compression in selected_ngrams[:100]:
        print(f"  {repr(ngram)}: freq={freq:,}, compression_value={compression:,.0f}")
    
    # Build vocabulary
    tokenizer.build_vocabulary(selected_ngrams)
    print(f"\nTotal vocabulary size: {len(tokenizer.vocab)}")
    
    return tokenizer, selected_ngrams


def test_tokenizer(tokenizer: SmartNgramTokenizer, test_texts: List[str]):
    """Test the tokenizer on sample texts."""
    print("\n" + "="*50)
    print("TOKENIZER TESTING")
    print("="*50)
    
    for text in test_texts:
        print(f"\nOriginal text: {repr(text)}")
        tokens = tokenizer.tokenize(text)
        print(f"Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        decoded = tokenizer.decode(tokens)
        print(f"Decoded: {repr(decoded)}")
        
        stats = tokenizer.analyze_compression(text)
        print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"Space savings: {stats['savings_percentage']:.1f}%")
        
        if stats['ngram_usage']:
            print("Top n-grams used:")
            for ngram, count in stats['ngram_usage'][:5]:
                print(f"  {repr(ngram)}: {count}x")


def main():
    # Base vocabulary
    base_vocab = [
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
    
    # Process Wikipedia data and create tokenizer

    tokenizer, selected_ngrams = process_and_select_ngrams(
        'scraped_data',  # Changed to folder path
        base_vocab,
        max_ngrams=10000,
        min_frequency=100
    )
    
    # Save vocabulary
    tokenizer.save_vocabulary('optimal_vocabulary.txt')
    
    # Save n-gram statistics
    with open('selected_ngrams.txt', 'w', encoding='utf-8') as f:
        f.write("Selected N-grams for Optimal Tokenization\n")
        f.write("="*50 + "\n\n")
        for i, (ngram, freq, compression) in enumerate(selected_ngrams):
            f.write(f"{i+1}. {repr(ngram)}: freq={freq:,}, compression_value={compression:,.0f}\n")
    
    print("Saving tokenizer as pickle file...")
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved as 'tokenizer.pkl'")
    
    # Test the tokenizer
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Pope Francis died on April 21, 2025.",
        "The Tiananmen Square protests of 1989.",
        "Hello World! This is a TEST of the tokenizer."
    ]
    
    test_tokenizer(tokenizer, test_texts)
    
    print(f"\n\nFinal vocabulary size: {len(tokenizer.vocab)} tokens")
    print("Base vocab size:", len(base_vocab))
    print("N-grams added:", len(tokenizer.vocab) - len(base_vocab))


if __name__ == "__main__":
    main()