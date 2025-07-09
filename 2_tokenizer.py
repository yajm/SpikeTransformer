import re
import pyphen
import pickle
import json
import os
from collections import Counter
from typing import List, Set, Tuple
import re
from tqdm import tqdm
import pyphen  # Install with: pip install pyphen

class ImprovedSyllableExtractor:
    def __init__(self):
        # Use pyphen for accurate syllabification
        self.dic = pyphen.Pyphen(lang='en')
        
        # Common prefixes and suffixes that should stay together
        self.common_affixes = {
            'prefixes': {'un', 're', 'pre', 'dis', 'over', 'under', 'mis', 'out', 
                        'anti', 'de', 'non', 'fore', 'inter', 'semi', 'super', 
                        'trans', 'ultra', 'bio', 'geo', 'micro', 'multi'},
            'suffixes': {'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ment',
                        'ness', 'ful', 'less', 'able', 'ible', 'ous', 'ive',
                        'ity', 'ize', 'ise', 'ate', 'fy', 'dom', 'ship', 'hood'}
        }
        
    def is_valid_syllable(self, syllable: str) -> bool:
        """Check if a string is a valid syllable"""
        # Must have at least one vowel
        if not any(c in 'aeiouy' for c in syllable.lower()):
            return False
        
        # Single letters are not syllables (except 'a' and 'I')
        if len(syllable) == 1 and syllable.lower() not in ['a', 'i']:
            return False
        
        # Must be pronounceable (basic check)
        if len(syllable) > 1:
            # Too many consonants in a row (more than 3)
            consonants = re.sub(r'[aeiouy]', '', syllable.lower())
            if len(consonants) > 3 and len(consonants) == len(syllable):
                return False
                
        return True
    
    def syllabify_word(self, word: str) -> List[str]:
        """Syllabify a single word using pyphen"""
        word = word.lower().strip()
        
        # Handle contractions
        if "'" in word:
            parts = word.split("'")
            if len(parts) == 2 and parts[1] in ['s', 't', 'd', 'll', 've', 're', 'm']:
                # Handle the main part
                syllables = self.syllabify_word(parts[0])
                # Add the contraction as a separate syllable
                syllables.append("'" + parts[1])
                return syllables
        
        # Use pyphen for syllabification
        syllabified = self.dic.inserted(word)
        syllables = syllabified.split('-')
        
        # Post-process to keep meaningful units together
        processed_syllables = []
        i = 0
        while i < len(syllables):
            current = syllables[i]
            
            # Check if we should merge with next syllable
            if i < len(syllables) - 1:
                combined = current + syllables[i + 1]
                
                # Keep common affixes together
                if (combined in self.common_affixes['prefixes'] or 
                    combined in self.common_affixes['suffixes']):
                    processed_syllables.append(combined)
                    i += 2
                    continue
                
                # Keep very short syllables together if they form common patterns
                if len(current) <= 2 and len(syllables[i + 1]) <= 2:
                    if self.is_valid_syllable(combined) and len(combined) <= 4:
                        processed_syllables.append(combined)
                        i += 2
                        continue
            
            processed_syllables.append(current)
            i += 1
        
        # Filter out invalid syllables
        return [s for s in processed_syllables if self.is_valid_syllable(s)]
    
    def extract_from_text(self, text: str) -> List[str]:
        """Extract all syllables from text"""
        # Clean text
        text = re.sub(r'[^\w\s\'-]', ' ', text.lower())
        words = text.split()
        
        all_syllables = []
        for word in words:
            # Skip numbers and very short words
            if word.isdigit() or len(word) < 2:
                continue
            
            # Handle hyphenated words
            if '-' in word and not word.startswith('-') and not word.endswith('-'):
                parts = word.split('-')
                for part in parts:
                    if part:
                        syllables = self.syllabify_word(part)
                        all_syllables.extend(syllables)
            else:
                syllables = self.syllabify_word(word)
                all_syllables.extend(syllables)
        
        return all_syllables

def analyze_syllable_distribution(syllable_counter: Counter, top_n: int = 100):
    """Analyze the distribution of syllables"""
    
    # Group by length
    length_dist = Counter()
    pattern_examples = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    
    for syllable, count in syllable_counter.most_common(1000):
        length = len(syllable)
        length_dist[length] += count
        
        if length in pattern_examples and len(pattern_examples[length]) < 10:
            pattern_examples[length].append((syllable, count))
    
    print("\nSyllable Length Distribution:")
    for length in sorted(length_dist.keys()):
        print(f"  {length} chars: {length_dist[length]:,} occurrences")
        if length in pattern_examples:
            examples = [f"'{s}' ({c:,})" for s, c in pattern_examples[length][:5]]
            print(f"    Examples: {', '.join(examples)}")
    
    # Analyze patterns
    print("\nCommon Patterns:")
    
    # Prefixes
    prefix_syllables = [(s, c) for s, c in syllable_counter.most_common(200) 
                       if s in ['un', 're', 'pre', 'dis', 'over', 'under', 'de', 'anti']]
    if prefix_syllables:
        print("  Prefixes:", ', '.join([f"'{s}' ({c:,})" for s, c in prefix_syllables[:8]]))
    
    # Suffixes
    suffix_syllables = [(s, c) for s, c in syllable_counter.most_common(200) 
                       if s in ['ing', 'ed', 'er', 'tion', 'ment', 'ness', 'ly', 'ful', 'able']]
    if suffix_syllables:
        print("  Suffixes:", ', '.join([f"'{s}' ({c:,})" for s, c in suffix_syllables[:8]]))
    
    # Common words as syllables
    word_syllables = [(s, c) for s, c in syllable_counter.most_common(100) 
                     if s in ['the', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'from', 'that']]
    if word_syllables:
        print("  Function words:", ', '.join([f"'{s}' ({c:,})" for s, c in word_syllables[:8]]))

def extract_syllables_from_wikipedia(folder_path: str, top_n: int = 2000):
    """Extract top N syllables from Wikipedia data using improved algorithm"""
    
    extractor = ImprovedSyllableExtractor()
    syllable_counter = Counter()
    
    # Also track full words to identify which syllables are complete words
    word_counter = Counter()
    
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files to process")
    
    total_articles = 0
    total_syllables = 0
    
    for json_file in tqdm(json_files, desc="Processing files"):
        file_path = os.path.join(folder_path, json_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            for article in articles:
                if 'content' in article and article['content']:
                    content = article['content'][:20000]  # Limit per article
                    
                    # Track words
                    words = re.findall(r'\b[a-z]+\b', content.lower())
                    word_counter.update(words)
                    
                    # Extract syllables
                    syllables = extractor.extract_from_text(content)
                    syllable_counter.update(syllables)
                    
                    total_articles += 1
                    total_syllables += len(syllables)
                    
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"\nProcessed {total_articles} articles")
    print(f"Total syllables found: {total_syllables}")
    print(f"Unique syllables: {len(syllable_counter)}")
    
    # Filter syllables
    filtered_syllables = []
    
    for syllable, count in syllable_counter.most_common(top_n * 2):  # Get extra to filter
        # Skip if it's a fragment that appears less than a threshold
        if count < 20:
            continue
            
        # Check if it's a complete word
        is_word = syllable in word_counter and word_counter[syllable] > 10
        
        # Keep if it's a common syllable or a complete word
        if count > 50 or is_word:
            filtered_syllables.append((syllable, count, is_word))
        
        if len(filtered_syllables) >= top_n:
            break
    
    # Analyze the distribution
    analyze_syllable_distribution(syllable_counter, top_n)
    
    return filtered_syllables, syllable_counter

def create_syllable_vocabulary(syllables: List[Tuple[str, int, bool]]):
    """Create vocabulary from syllables"""
    
    # Base tokens
    base_tokens = [
        '<PAD>', '<START>', '<END>', '<UNK>', '<SPACE>', '<CAP>',
        # Single-character tokens for fallback
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        # Numbers
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        # Punctuation
        '.', ',', '!', '?', '-', "'", '"', ':', ';', '(', ')', '\n', '–', '/', '“', '”', '$', '=',
        ']', '[', '+', '}', '{', '&', '%', '#', '—'
    ]
    
    vocab = base_tokens.copy()
    
    print("\nTop 50 syllables:")
    for i, (syllable, count, is_word) in enumerate(syllables[:50]):
        word_marker = " [WORD]" if is_word else ""
        print(f"{i+1}. '{syllable}': {count:,}{word_marker}")
    
    # Add syllables to vocabulary
    for syllable, count, is_word in syllables:
        if syllable not in vocab:  # Avoid duplicates with base tokens
            vocab.append(syllable)
    
    print(f"\nFinal vocabulary size: {len(vocab)}")
    print(f"  Base tokens: {len(base_tokens)}")
    print(f"  Syllable tokens: {len(vocab) - len(base_tokens)}")
    
    return vocab

class SyllableTokenizer:
    """Simple syllable tokenizer compatible with SpikeLLM"""
    
    def __init__(self, vocab_file='syllable_vocabulary_improved.txt'):
        # Load vocabulary from file
        self.vocab = []
        self.vocab_to_id = {}
        self.id_to_vocab = {}
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                # Don't strip before splitting - just remove the trailing newline
                line = line.rstrip('\n')
                parts = line.split('\t')
                if len(parts) == 2:
                    idx, token = int(parts[0]), parts[1]
                    # Unescape special characters
                    if token == '\\n':
                        token = '\n'
                    elif token == '\\t':
                        token = '\t'
                    self.vocab.append(token)
                    self.vocab_to_id[token] = idx
                    self.id_to_vocab[idx] = token
        
        # Initialize pyphen for syllabification
        self.dic = pyphen.Pyphen(lang='en')
        
        # Special token IDs
        self.pad_id = self.vocab_to_id['<PAD>']
        self.unk_id = self.vocab_to_id['<UNK>']
        self.space_id = self.vocab_to_id['<SPACE>']
        self.cap_id = self.vocab_to_id['<CAP>']
    
        
    def tokenize(self, text):
        """Convert text to token IDs - completely skip unknown characters"""
        tokens = []
        
        # Split by spaces but keep track of them
        text = text.replace('\n', ' \n ')
        words = text.split(' ')
        
        for i, word in enumerate(words):
            if not word:
                continue
                
            # Add space before word (except first)
            if i > 0 and '<SPACE>' in self.vocab_to_id:
                tokens.append(self.space_id)
            
            # Handle capitalization
            is_capitalized = word and word[0].isupper()
            if is_capitalized and '<CAP>' in self.vocab_to_id:
                tokens.append(self.cap_id)
                word = word.lower()
            
            # Extract punctuation
            prefix_punct = ''
            suffix_punct = ''
            
            # Get leading punctuation
            while word and not word[0].isalnum():
                prefix_punct += word[0]
                word = word[1:]
            
            # Get trailing punctuation
            while word and not word[-1].isalnum():
                suffix_punct = word[-1] + suffix_punct
                word = word[:-1]
            
            # Add prefix punctuation tokens - ONLY if they exist in vocab
            for char in prefix_punct:
                if char in self.vocab_to_id:
                    tokens.append(self.vocab_to_id[char])
                # If not in vocab, completely skip it (no pass, no append)
            
            # Tokenize the word
            if word:
                # First check if the whole word is in vocab
                if word in self.vocab_to_id:
                    tokens.append(self.vocab_to_id[word])
                else:
                    # Try to use syllables when possible
                    try:
                        syllables = self.dic.inserted(word).split('-')
                        
                        # Process each syllable
                        for syllable in syllables:
                            if syllable in self.vocab_to_id:
                                # Syllable is in vocab, use it
                                tokens.append(self.vocab_to_id[syllable])
                            else:
                                # Syllable not in vocab, try individual characters
                                for char in syllable:
                                    if char in self.vocab_to_id:
                                        tokens.append(self.vocab_to_id[char])
                                    # If char not in vocab, completely skip it
                    except:
                        # If syllabification fails, try character by character
                        for char in word:
                            if char in self.vocab_to_id:
                                tokens.append(self.vocab_to_id[char])
                            # If char not in vocab, completely skip it
            
            # Add suffix punctuation tokens - ONLY if they exist in vocab
            for char in suffix_punct:
                if char in self.vocab_to_id:
                    tokens.append(self.vocab_to_id[char])
                # If not in vocab, completely skip it
        
        return tokens
    
    def tokenize_debug(self, text):
        """Debug version that shows tokenization process"""
        tokens = []
        debug_info = []
        
        words = text.split(' ')
        
        for i, word in enumerate(words):
            if not word:
                continue
                
            if i > 0:
                tokens.append(self.space_id)
                debug_info.append(('<SPACE>', self.space_id))
            
            # Handle capitalization
            is_capitalized = word and word[0].isupper()
            if is_capitalized:
                tokens.append(self.cap_id)
                debug_info.append(('<CAP>', self.cap_id))
                word = word.lower()
            
            # Extract punctuation
            prefix_punct = ''
            suffix_punct = ''
            while word and not word[0].isalnum():
                prefix_punct += word[0]
                word = word[1:]
            while word and not word[-1].isalnum():
                suffix_punct = word[-1] + suffix_punct
                word = word[:-1]
            
            for char in prefix_punct:
                if char in self.vocab_to_id:
                    tokens.append(self.vocab_to_id[char])
                    debug_info.append((char, self.vocab_to_id[char]))
            
            if word:
                if word in self.vocab_to_id:
                    tokens.append(self.vocab_to_id[word])
                    debug_info.append((f"WORD:{word}", self.vocab_to_id[word]))
                else:
                    syllables = self.dic.inserted(word).split('-')
                    debug_info.append((f"SYLLABLES:{word}→{syllables}", None))
                    
                    for syllable in syllables:
                        if syllable in self.vocab_to_id:
                            tokens.append(self.vocab_to_id[syllable])
                            debug_info.append((f"  SYL:{syllable}", self.vocab_to_id[syllable]))
                        else:
                            debug_info.append((f"  NO_SYL:{syllable}→chars", None))
                            for char in syllable:
                                if char in self.vocab_to_id:
                                    tokens.append(self.vocab_to_id[char])
                                    debug_info.append((f"    CHAR:{char}", self.vocab_to_id[char]))
            
            for char in suffix_punct:
                if char in self.vocab_to_id:
                    tokens.append(self.vocab_to_id[char])
                    debug_info.append((char, self.vocab_to_id[char]))
        
        return tokens, debug_info
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        text_parts = []
        skip_next_space = True
        next_cap = False
        
        for token_id in token_ids:
            if token_id == self.pad_id:
                continue
            
            token = self.id_to_vocab.get(token_id, '<UNK>')
            
            if token == '<START>' or token == '<END>':
                continue
            elif token == '<CAP>':
                next_cap = True
            elif token == '<SPACE>':
                if not skip_next_space:
                    text_parts.append(' ')
                skip_next_space = False
            else:
                if token in '.,!?;:\'"()':
                    # Punctuation - no space before
                    skip_next_space = True
                
                if next_cap and token.isalpha():
                    text_parts.append(token[0].upper() + token[1:])
                    next_cap = False
                else:
                    text_parts.append(token)
                    
                skip_next_space = False
        
        return ''.join(text_parts)
    
    def save(self, filepath):
        """Save tokenizer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Example usage
if __name__ == "__main__":
    # Create and save the tokenizer

    folder_path = 'scraped_data'
    
    print("Extracting syllables with improved algorithm...")
    syllables, syllable_counter = extract_syllables_from_wikipedia(folder_path, top_n=2000)
    
    # Create vocabulary
    vocab = create_syllable_vocabulary(syllables)
    
    # Save vocabulary
    with open('syllable_vocabulary_improved.txt', 'w', encoding='utf-8') as f:
        f.write("# Syllable-based Vocabulary (Improved)\n")
        f.write(f"# Total tokens: {len(vocab)}\n")
        f.write("#\n")
        for i, token in enumerate(vocab):
            # Escape special characters
            if token == '\n':
                f.write(f"{i}\t\\n\n")  # Write as literal \n
            elif token == '\t':
                f.write(f"{i}\t\\t\n")  # Write as literal \t
            else:
                f.write(f"{i}\t{token}\n")

    tokenizer = SyllableTokenizer('syllable_vocabulary_improved.txt')
    tokenizer.save('syllable_tokenizer.pkl')
    
    # Test it
    test_text = "The Oklahoma City Thunder are an American professional basketball team based in Oklahoma City."
    tokens = tokenizer.tokenize(test_text)
    print(f"Text: {test_text}")
    print(len(test_text))
    print(f"Tokens: {tokens}")
    print(len(tokens))
    print(len(tokens)/len(test_text))
    print(f"Decoded: {tokenizer.decode(tokens)}")