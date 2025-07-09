import os
import json
import re
from anthropic import Anthropic
from typing import List, Dict

client = Anthropic(
    api_key=""
)

def split_content_into_sections(content: str, num_sections: int = 5) -> List[str]:
    """Split content into equal-length sections."""
    # Clean the content
    content = content.strip()
    
    # Calculate approximate length per section
    total_length = len(content)
    section_length = total_length // num_sections
    
    sections = []
    for i in range(num_sections):
        start_idx = i * section_length
        if i == num_sections - 1:
            # Last section gets any remaining content
            end_idx = total_length
        else:
            end_idx = (i + 1) * section_length
            # Try to find a sentence boundary near the end_idx
            search_start = max(0, end_idx - 100)
            search_end = min(total_length, end_idx + 100)
            
            # Look for sentence endings
            sentence_endings = ['.', '!', '?']
            best_idx = end_idx
            
            for j in range(search_start, search_end):
                if j < total_length and content[j] in sentence_endings:
                    # Check if it's not an abbreviation
                    if j + 1 < total_length and content[j + 1] == ' ':
                        best_idx = j + 1
                        break
            
            end_idx = best_idx
        
        section = content[start_idx:end_idx].strip()
        sections.append(section)
    
    return sections

def generate_qa_for_section(article_title: str, section_content: str, section_id: int, category: str) -> Dict:
    """Generate a Q&A pair for a given section using Claude API."""
    
    prompt = f"""Based on the following section from an article titled "{article_title}" in the category "{category}", create ONE question and answer pair.

Section {section_id} content:
{section_content}

Requirements:
1. The question must be fully answerable using ONLY the information in this section
2. The answer should be as simple and concise as possible
3. Use direct facts from the text
4. Format the answer like: "June 6, 1980" not "Peter Brian was born on June 6, 1980"

Return your response in this exact JSON format:
{{
    "question": "Your question here",
    "answer": "Your concise answer here"
}}

Only return the JSON, no other text."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",  # Using Sonnet for cost efficiency
            max_tokens=200,
            temperature=0.3,  # Lower temperature for more consistent outputs
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response
        response_text = message.content[0].text.strip()
        
        # Extract JSON from response
        try:
            qa_data = json.loads(response_text)
            return {
                "question": qa_data.get("question", ""),
                "answer": qa_data.get("answer", "")
            }
        except json.JSONDecodeError:
            # Try to extract JSON from the response if it's wrapped in other text
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                qa_data = json.loads(json_match.group())
                return {
                    "question": qa_data.get("question", ""),
                    "answer": qa_data.get("answer", "")
                }
            else:
                print(f"Failed to parse JSON from response: {response_text}")
                return None
                
    except Exception as e:
        print(f"Error generating Q&A: {str(e)}")
        return None

def process_articles(folder_path: str, output_file: str = "qa_pairs.json"):
    """Process all JSON files in the folder and generate Q&A pairs."""
    
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files to process")
    
    all_qa_pairs = []
    
    for file_idx, json_file in enumerate(json_files):
        file_path = os.path.join(folder_path, json_file)
        print(f"\nProcessing {json_file} ({file_idx + 1}/{len(json_files)})...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            
            for article in articles:
                article_id = article.get('rank', 0)
                title = article.get('title', '')
                category = article.get('category', '')
                content = article.get('content', '')
                
                print(f"  Processing article {article_id}: {title}")
                
                # Split content into 5 sections
                sections = split_content_into_sections(content, 5)
                
                # Generate Q&A for each section
                for section_id, section_content in enumerate(sections, 1):
                    if not section_content.strip():
                        continue
                    
                    print(f"    Generating Q&A for section {section_id}...")
                    
                    qa_pair = generate_qa_for_section(title, section_content, section_id, category)
                    
                    if qa_pair:
                        qa_entry = {
                            "category_id": category,
                            "article_id": article_id,
                            "section_id": section_id,
                            "question": qa_pair["question"],
                            "answer": qa_pair["answer"]
                        }
                        all_qa_pairs.append(qa_entry)
                        
                        # Print progress
                        print(f"      Q: {qa_pair['question']}")
                        print(f"      A: {qa_pair['answer']}")
                    
    
    # Save all Q&A pairs to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(all_qa_pairs)} Q&A pairs")
    print(f"Results saved to {output_file}")
    
    return all_qa_pairs

def main():
    folder_path = "scraped_data" 
    qa_pairs = process_articles(folder_path)
    
    # Print summary statistics
    if qa_pairs:
        categories = {}
        for qa in qa_pairs:
            cat = qa['category_id']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nSummary by category:")
        for cat, count in categories.items():
            print(f"  {cat}: {count} Q&A pairs")

if __name__ == "__main__":
    main()