import json
import pickle

from tokenizer import SmartNgramTokenizer

def analyze_qa_token_lengths(json_file='qa_pairs.json', tokenizer_file='tokenizer.pkl'):
    """
    Read Q&A pairs from JSON, tokenize them, and analyze token lengths.
    """
    # Load the tokenizer
    print("Loading tokenizer...")
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load the Q&A pairs
    print(f"Loading Q&A pairs from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f"Total Q&A pairs: {len(qa_pairs)}")
    
    # Tokenize and analyze
    token_lengths = []
    under_64 = 0
    under_96 = 0
    under_128 = 0
    
    # For debugging - track some examples
    examples_64 = []
    examples_96 = []
    examples_128 = []
    
    for i, qa in enumerate(qa_pairs):
        question = qa['question']
        answer = qa['answer']
        
        # Tokenize question and answer
        q_tokens = tokenizer.tokenize(question)
        a_tokens = tokenizer.tokenize(answer)
        
        # Combined length
        combined_length = len(q_tokens) + len(a_tokens)
        token_lengths.append(combined_length)
        
        # Count thresholds
        if combined_length < 64:
            under_64 += 1
            if len(examples_64) < 3:  # Save a few examples
                examples_64.append({
                    'question': question,
                    'answer': answer,
                    'q_tokens': len(q_tokens),
                    'a_tokens': len(a_tokens),
                    'total': combined_length
                })
        
        if combined_length < 96:
            under_96 += 1
            if len(examples_96) < 3 and combined_length >= 64:
                examples_96.append({
                    'question': question,
                    'answer': answer,
                    'q_tokens': len(q_tokens),
                    'a_tokens': len(a_tokens),
                    'total': combined_length
                })
        
        if combined_length < 128:
            under_128 += 1
            if len(examples_128) < 3 and combined_length >= 96:
                examples_128.append({
                    'question': question,
                    'answer': answer,
                    'q_tokens': len(q_tokens),
                    'a_tokens': len(a_tokens),
                    'total': combined_length
                })
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} Q&A pairs...")
    
    # Calculate statistics
    avg_length = sum(token_lengths) / len(token_lengths)
    min_length = min(token_lengths)
    max_length = max(token_lengths)
    
    # Print results
    print("\n" + "="*60)
    print("TOKEN LENGTH ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nTotal Q&A pairs analyzed: {len(qa_pairs)}")
    print(f"Average combined token length: {avg_length:.2f}")
    print(f"Min token length: {min_length}")
    print(f"Max token length: {max_length}")
    
    print(f"\nThreshold Analysis:")
    print(f"  < 64 tokens:  {under_64:,} pairs ({under_64/len(qa_pairs)*100:.1f}%)")
    print(f"  < 96 tokens:  {under_96:,} pairs ({under_96/len(qa_pairs)*100:.1f}%)")
    print(f"  < 128 tokens: {under_128:,} pairs ({under_128/len(qa_pairs)*100:.1f}%)")
    
    # Distribution analysis
    print(f"\nToken Length Distribution:")
    length_ranges = [(0, 32), (32, 64), (64, 96), (96, 128), (128, 160), (160, 200), (200, float('inf'))]
    for start, end in length_ranges:
        count = sum(1 for l in token_lengths if start <= l < end)
        if count > 0:
            label = f"{start}-{end-1}" if end != float('inf') else f"{start}+"
            print(f"  {label:>8} tokens: {count:,} pairs ({count/len(qa_pairs)*100:.1f}%)")
    
    # Show some examples
    print("\n" + "="*60)
    print("EXAMPLES")
    print("="*60)
    
    if examples_64:
        print("\nExamples under 64 tokens:")
        for ex in examples_64[:2]:
            print(f"  Q: {ex['question'][:60]}...")
            print(f"  A: {ex['answer']}")
            print(f"  Tokens: Q={ex['q_tokens']}, A={ex['a_tokens']}, Total={ex['total']}\n")
    
    # Additional analysis - compression efficiency
    print("\n" + "="*60)
    print("COMPRESSION ANALYSIS")
    print("="*60)
    
    total_chars = 0
    total_tokens = 0
    
    for qa in qa_pairs[:1000]:  # Sample first 1000 for speed
        q_chars = len(qa['question'])
        a_chars = len(qa['answer'])
        q_tokens = tokenizer.tokenize(qa['question'])
        a_tokens = tokenizer.tokenize(qa['answer'])
        
        total_chars += q_chars + a_chars
        total_tokens += len(q_tokens) + len(a_tokens)
    
    print(f"\nSample compression analysis (first 1000 Q&A pairs):")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Compression ratio: {total_chars/total_tokens:.2f}x")
    print(f"  Space savings: {(1 - total_tokens/total_chars)*100:.1f}%")
    
    return {
        'total_pairs': len(qa_pairs),
        'under_64': under_64,
        'under_96': under_96,
        'under_128': under_128,
        'avg_length': avg_length,
        'token_lengths': token_lengths
    }

if __name__ == "__main__":
    # Run the analysis
    results = analyze_qa_token_lengths()
    
    # Optional: Save detailed results
    with open('qa_token_analysis.json', 'w') as f:
        json.dump({
            'total_pairs': results['total_pairs'],
            'under_64': results['under_64'],
            'under_96': results['under_96'],
            'under_128': results['under_128'],
            'avg_length': results['avg_length']
        }, f, indent=2)
    
    print("\nResults saved to 'qa_token_analysis.json'")