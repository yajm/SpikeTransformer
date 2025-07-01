import json
from collections import Counter

def analyze_character_frequency(filename='wikipedia_top_articles.json'):
    """
    Analyze character frequency in Wikipedia articles
    
    Args:
        filename: Path to the JSON file containing Wikipedia articles
    
    Returns:
        tuple: (character_counter, total_unique_chars)
    """
    
    # Read the JSON file
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize a counter for all characters
    char_counter = Counter()
    
    # Process each article
    for article in data:
        # Get the content of the article
        content = article.get('content', '')
        
        # Count each character in the content
        for char in content:
            char_counter[char] += 1
    
    # Get total number of unique characters
    total_unique_chars = len(char_counter)
    
    # Get top 100 most common characters
    top_100_chars = char_counter.most_common(100)
    
    return char_counter, total_unique_chars, top_100_chars

def display_results(char_counter, total_unique_chars, top_100_chars):
    """
    Display the analysis results
    """
    print("=" * 80)
    print("CHARACTER FREQUENCY ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal number of different characters: {total_unique_chars}")
    
    print("\n" + "=" * 80)
    print("TOP 100 MOST FREQUENT CHARACTERS")
    print("=" * 80)
    
    # Calculate total count of top 100 characters
    top_100_total = sum(count for _, count in top_100_chars)
    
    print(f"\nTotal count of top 100 characters: {top_100_total:,}")
    print("\n{:<5} {:<15} {:<15} {:<20}".format("Rank", "Character", "Count", "Representation"))
    print("-" * 60)
    
    for i, (char, count) in enumerate(top_100_chars, 1):
        # Create a readable representation of the character
        if char == ' ':
            char_repr = '[SPACE]'
        elif char == '\n':
            char_repr = '[NEWLINE]'
        elif char == '\t':
            char_repr = '[TAB]'
        elif char == '\r':
            char_repr = '[RETURN]'
        elif ord(char) < 32 or ord(char) > 126:
            char_repr = f'[U+{ord(char):04X}]'
        else:
            char_repr = char
            
        print(f"{i:<5} {char_repr:<15} {count:<15,} {repr(char):<20}")
    
    # Additional statistics
    print("\n" + "=" * 80)
    print("ADDITIONAL STATISTICS")
    print("=" * 80)
    
    # Total characters
    total_chars = sum(char_counter.values())
    print(f"\nTotal number of characters: {total_chars:,}")
    
    # Character categories
    ascii_chars = sum(1 for char in char_counter if ord(char) < 128)
    non_ascii_chars = total_unique_chars - ascii_chars
    
    print(f"ASCII characters: {ascii_chars}")
    print(f"Non-ASCII characters: {non_ascii_chars}")
    
    # Most common non-ASCII characters
    print("\nTop 10 non-ASCII characters:")
    non_ascii_top = [(char, count) for char, count in char_counter.most_common() 
                     if ord(char) >= 128][:10]
    
    for i, (char, count) in enumerate(non_ascii_top, 1):
        print(f"{i}. {repr(char)} (U+{ord(char):04X}): {count:,}")

def save_full_analysis(char_counter, filename='character_frequency_full.txt'):
    """
    Save the complete character frequency analysis to a file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Complete Character Frequency Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total unique characters: {len(char_counter)}\n")
        f.write(f"Total character count: {sum(char_counter.values()):,}\n\n")
        
        f.write("All characters sorted by frequency:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Rank':<6} {'Character':<20} {'Count':<15} {'Unicode':<10}\n")
        f.write("-" * 60 + "\n")
        
        for i, (char, count) in enumerate(char_counter.most_common(), 1):
            if char == ' ':
                char_repr = '[SPACE]'
            elif char == '\n':
                char_repr = '[NEWLINE]'
            elif char == '\t':
                char_repr = '[TAB]'
            elif char == '\r':
                char_repr = '[RETURN]'
            elif ord(char) < 32 or ord(char) > 126:
                char_repr = f'[U+{ord(char):04X}]'
            else:
                char_repr = repr(char)
                
            f.write(f"{i:<6} {char_repr:<20} {count:<15,} U+{ord(char):04X}\n")
    
    print(f"\nFull analysis saved to: {filename}")

# Main execution
if __name__ == "__main__":
    try:
        # Analyze the character frequency
        char_counter, total_unique_chars, top_100_chars = analyze_character_frequency()
        
        # Display the results
        display_results(char_counter, total_unique_chars, top_100_chars)
        
        # Save full analysis to file
        save_full_analysis(char_counter)
        
    except FileNotFoundError:
        print("Error: Could not find 'wikipedia_top_articles.json' file.")
        print("Please make sure the file is in the same directory as this script.")
    except json.JSONDecodeError:
        print("Error: The JSON file is not properly formatted.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")