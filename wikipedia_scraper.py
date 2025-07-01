import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict
import time
import wikipediaapi

def get_wikipedia_article_content(title):
    """
    Get the full text content of a Wikipedia article using wikipedia-api
    """
    try:
        # Initialize Wikipedia API with a custom user agent
        wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='WikiStats Bot/1.0 (https://example.com/contact)'
        )
        
        # Get the page
        page = wiki.page(title)
        
        if page.exists():
            # Return the full text content
            return page.text
        else:
            print(f"Page not found: {title}")
            return None
            
    except Exception as e:
        print(f"Error fetching content for {title}: {e}")
        return None

def get_alltime_top_articles(sample_days=100, top_n=1000):
    """
    Get all-time most popular Wikipedia articles by sampling historical data
    """
    print(f"Collecting all-time most popular Wikipedia articles...")
    print(f"Sampling {sample_days} days of historical data...")
    
    # Dictionary to store cumulative views
    article_total_views = defaultdict(int)
    article_appearances = defaultdict(int)
    
    # Sample evenly across the time period
    for i in range(0, sample_days, 7):  # Sample weekly to speed up
        date = datetime.now() - timedelta(days=i+1)
        date_str = date.strftime('%Y/%m/%d')
        
        try:
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{date_str}"
            response = requests.get(url, headers={'User-Agent': 'WikiStats Bot/1.0'})
            
            if response.status_code == 200:
                data = response.json()
                articles = data['items'][0]['articles']
                
                # Process all articles from this day
                for article in articles:
                    # Skip special pages and main page
                    if not any(prefix in article['article'] for prefix in 
                              ['Special:', 'Main_Page', 'Portal:', 'File:', '-', 'Wikipedia:', 'Help:']):
                        article_total_views[article['article']] += article['views']
                        article_appearances[article['article']] += 1
                
                if (i // 7) % 4 == 0:  # Progress update every 4 weeks
                    print(f"Processed {i} days...")
                
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
    
    print("\nCalculating all-time rankings...")
    
    # Calculate metrics for each article
    article_stats = []
    for article, total_views in article_total_views.items():
        appearances = article_appearances[article]
        if appearances > 0:
            article_stats.append({
                'title': article.replace('_', ' '),
                'title_encoded': article,  # Keep the URL-encoded version
                'total_views': total_views,
                'appearances': appearances,
                'avg_daily_views': total_views // appearances
            })
    
    # Sort by total views
    article_stats.sort(key=lambda x: x['total_views'], reverse=True)
    
    return article_stats[:top_n]

def scrape_articles_with_content(articles, limit=None):
    """
    Fetch the actual content for each article
    """
    print(f"\n📚 Fetching article content...")
    print(f"This will take some time. Processing {limit or len(articles)} articles...")
    
    results = []
    
    for i, article in enumerate(articles[:limit] if limit else articles):
        if i % 10 == 0:
            print(f"Progress: {i}/{limit or len(articles)} articles fetched...")
        
        # Use the title with spaces for wikipedia-api
        content = get_wikipedia_article_content(article['title'])
        
        if content:
            # Get content length in words
            word_count = len(content.split())
            
            results.append({
                'rank': i + 1,
                'title': article['title'],
                'view_count': article['total_views'],
                'content': content,
                'word_count': word_count
            })
            
            # Show progress with word count
            if (i + 1) % 5 == 0:
                print(f"  ✓ {article['title']} ({word_count:,} words)")
        else:
            print(f"  ⚠️  Failed to fetch: {article['title']}")
        
        # Rate limiting - be respectful to Wikipedia's servers
        time.sleep(0.3)
    
    return results

def save_results_json(articles_with_content, filename='wikipedia_top_articles.json'):
    """Save results to JSON file"""
    import os
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(articles_with_content, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Successfully saved {len(articles_with_content)} articles to '{filename}'")
    
    # Show summary
    print("\n📊 SUMMARY:")
    print(f"- Total articles saved: {len(articles_with_content)}")
    print(f"- File size: {os.path.getsize(filename) / 1024 / 1024:.1f} MB")
    
    # Calculate total words
    total_words = sum(article['word_count'] for article in articles_with_content)
    avg_words = total_words // len(articles_with_content) if articles_with_content else 0
    
    print(f"- Total words: {total_words:,}")
    print(f"- Average words per article: {avg_words:,}")
    
    print("\n🏆 TOP 10 ARTICLES:")
    for article in articles_with_content[:10]:
        content_preview = ' '.join(article['content'].split()[:20]) + '...'
        print(f"{article['rank']:3d}. {article['title']:<30} | {article['view_count']:>10,} views | {article['word_count']:>6,} words")
        print(f"     {content_preview}\n")

def main():
    """Main execution"""
    import os
    
    print("🔍 Wikipedia Top Articles Scraper (with FULL Content)")
    print("=" * 60)
    
    # Check if wikipedia-api is installed
    try:
        import wikipediaapi
    except ImportError:
        print("\n⚠️  Please install wikipedia-api first:")
        print("    pip install wikipedia-api")
        return
    
    # First, get the top articles by view count
    articles = get_alltime_top_articles(sample_days=100, top_n=1000)
    
    if not articles:
        print("❌ Failed to retrieve top articles")
        return
    
    # Ask user how many articles to fetch content for
    print(f"\n📋 Found {len(articles)} top articles")
    print("\n⚠️  WARNING: Fetching content for all 1000 articles will:")
    print("   - Take approximately 5-10 minutes")
    print("   - Create a JSON file of 100-300 MB")
    print("   - Download the COMPLETE content of each article")
    
    # For this example, let's fetch content for top 50 articles
    # You can change this to fetch all 1000 if needed
    FETCH_LIMIT = 1000  # Change this to 1000 to fetch all
    
    print(f"\n🚀 Fetching FULL content for top {FETCH_LIMIT} articles...")
    
    # Fetch article content
    articles_with_content = scrape_articles_with_content(articles, limit=FETCH_LIMIT)
    
    # Save to JSON
    save_results_json(articles_with_content)
    
    print("\n💡 USAGE TIPS:")
    print("- To fetch all 1000 articles, change FETCH_LIMIT to 1000")
    print("- The JSON structure includes: rank, title, view_count, content, word_count")
    print("- Content includes the COMPLETE Wikipedia article text")
    print("- Consider using gzip compression for storage")
    print("\n📝 Example of loading the data:")
    print("    import json")
    print("    with open('wikipedia_top_articles.json', 'r') as f:")
    print("        articles = json.load(f)")
    print("    print(articles[0]['content'])  # Full text of top article")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total processing time: {elapsed/60:.1f} minutes")