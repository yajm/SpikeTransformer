import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict
import time
import wikipediaapi
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq

class OptimizedWikipediaScraper:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='WikiStats Bot/1.0 (https://example.com/contact)'
        )
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'WikiStats Bot/1.0'})
        
    def get_top_articles_by_views(self, days=30, limit=1000):
        """
        Get top articles directly from Wikipedia's most viewed pages API
        This is MUCH faster than scanning categories
        """
        print(f"Fetching top {limit} most viewed articles from the last {days} days...")
        
        all_articles = []
        end_date = datetime.now()
        
        # Wikipedia provides daily top article lists, we'll aggregate them
        for day_offset in range(min(days, 30)):  # API limits to ~30 days
            date = end_date - timedelta(days=day_offset)
            date_str = date.strftime('%Y/%m/%d')
            
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{date_str}"
            
            try:
                response = self.session.get(url)
                if response.status_code == 200:
                    data = response.json()
                    for article in data['items'][0]['articles']:
                        all_articles.append({
                            'title': article['article'],
                            'views': article['views'],
                            'date': date_str
                        })
                # time.sleep(0.1)
            except Exception as e:
                print(f"Error fetching top articles for {date_str}: {e}")
        
        # Aggregate views by article
        article_views = defaultdict(int)
        for item in all_articles:
            article_views[item['title']] += item['views']
        
        # Sort and return top articles
        sorted_articles = sorted(
            article_views.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return sorted_articles

    def get_category_top_viewed(self, category_name, limit=100, sample_size=5000):
        """
        Smart category scraping - only check a sample of articles for views
        instead of trying to get ALL articles
        """
        print(f"\nProcessing category: {category_name}")
        
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': f'Category:{category_name}',
            'cmlimit': sample_size,  # Only get a sample
            'cmtype': 'page',
            'format': 'json'
        }
        
        articles = []
        
        try:
            response = self.session.get(url, params=params)
            data = response.json()
            
            if 'query' in data:
                for member in data['query']['categorymembers']:
                    if member['ns'] == 0:
                        articles.append(member['title'])
                        
        except Exception as e:
            print(f"Error fetching category {category_name}: {e}")
            return []
        
        # Get views for the sample
        print(f"  Checking views for {len(articles)} sample articles...")
        article_views = self.get_article_views_batch(articles[:sample_size], days=7)  # Only check 7 days for speed
        
        # Return top viewed from the sample
        sorted_articles = sorted(
            [(title, views) for title, views in article_views.items() if views > 0],
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return sorted_articles

    def get_smart_famous_people(self, limit=100):
        """
        Get famous people using smarter subcategories that are more manageable
        """
        # Use more specific categories that are likely to contain famous people
        smart_categories = [
            'Nobel laureates',
            'Academy Award winners', 
            'Grammy Award winners',
            'Olympic medalists',
            'Heads of state',
            'Billionaires',
            'Film directors',
            'Best-selling music artists',
            '21st-century American politicians',
            'American film actors',
            'English footballers',
            'NBA players',
            'Scientists who committed suicide',
            'People from New York City',
            'Harvard University alumni',
            'Recipients of the Presidential Medal of Freedom'
        ]
        
        all_people = {}
        
        print("\n🎯 Smart Famous People Search")
        print("Using targeted subcategories instead of broad categories...")
        
        for category in smart_categories:
            print(f"\n  Checking: {category}")
            top_in_category = self.get_category_top_viewed(category, limit=20, sample_size=1000)
            
            for title, views in top_in_category:
                if title not in all_people or all_people[title] < views:
                    all_people[title] = views
        
        # Sort all collected people by views
        sorted_people = sorted(
            all_people.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return sorted_people

    def get_article_views_batch(self, articles, days=30):
        """
        Optimized batch view fetching with better error handling
        """
        article_views = {}
        
        def fetch_views(article):
            total_views = 0
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            
            encoded_title = article.replace(' ', '_')
            
            try:
                # Use daily endpoint for shorter periods, monthly for longer
                if days <= 30:
                    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{encoded_title}/daily/{start_str}/{end_str}"
                else:
                    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{encoded_title}/monthly/{start_str}/{end_str}"
                
                response = self.session.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data['items']:
                        total_views += item['views']
                        
            except Exception:
                pass  # Silently skip errors for efficiency
            
            return article, total_views
        
        # Use thread pool with reasonable limits
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(fetch_views, article) for article in articles]
            
            for future in as_completed(futures):
                try:
                    article, views = future.result()
                    article_views[article] = views
                except Exception:
                    pass
        
        return article_views

    def get_wikipedia_article_content(self, title):
        """Get article content"""
        try:
            page = self.wiki.page(title)
            if page.exists():
                return page.text
            return None
        except Exception:
            return None

    def smart_category_scrape(self):
        """
        Main smart scraping function that combines multiple strategies
        """
        results = {}
        
        # Strategy 1: Get globally most viewed articles
        print("\n📊 Strategy 1: Fetching globally most viewed articles...")
        top_global = self.get_top_articles_by_views(days=7, limit=1000)
        """
        # Strategy 2: Get famous people using smart subcategories
        print("\n👥 Strategy 2: Fetching famous people from targeted categories...")
        famous_people = self.get_smart_famous_people(limit=1000)
        
        # Strategy 3: Use specific high-quality categories for each domain
        domain_categories = {
            'Biology': ['Cell biology', 'Genetics', 'Evolution', 'Human biology'],
            'Physics': ['Quantum mechanics', 'Relativity', 'Particle physics', 'Astrophysics'],
            'History': ['World War II', 'Ancient Rome', 'American Civil War', 'Medieval history'],
            'Chemistry': ['Chemical elements', 'Organic compounds', 'Nobel laureates in Chemistry'],
            'Mathematics': ['Mathematical theorems', 'Fields Medal winners', 'Number theory'],
            'Geography': ['Capital cities', 'World Heritage Sites', 'Mountain ranges', 'Rivers']
        }
        """

        domain_categories = {
    'Cities': [
        'Capitals in Europe',
        'Capitals in Asia', 
        'Capitals in Africa',
        'Cities in the United States by state',
        'Million-plus agglomerations in India',
        'Megacities'
    ],
    'Companies': [
        'Companies in the Dow Jones Industrial Average',
        'S&P 100',
        'Technology companies established in the 21st century',
        'Automotive companies of Japan',
        'Banks of the United States'
    ]
}
        """
            'Companies': ['S&P 500 companies', 'Tech startups', 'Fortune 500', 'Unicorn companies', 'Multinational corporations', 'FAANG companies', 'Retail giants', 'Pharmaceutical companies', 'Automotive manufacturers', 'Financial institutions'],
                'Computing': ['Programming languages', 'Operating systems', 'Computer scientists', 'Algorithms', 'Cybersecurity'],
        'Religion': ['World religions', 'Religious texts', 'Religious leaders', 'Sacred sites', 'Religious holidays'],
        'Cities': ['World capitals', 'Megacities', 'Historic cities', 'Port cities', 'Tech hubs'],
        'Literature': ['Classic novels', 'Poetry', 'Literary awards', 'Shakespeare', 'Science fiction'],
        'Art': ['Famous paintings', 'Art movements', 'Museums', 'Sculptors', 'Renaissance artists'],
        'Music': ['Classical composers', 'Music genres', 'Musical instruments', 'Opera', 'Grammy winners'],
        'Philosophy': ['Ancient philosophers', 'Ethical theories', 'Logic', 'Existentialism', 'Political philosophy'],
        'Psychology': ['Cognitive biases', 'Psychological disorders', 'Famous psychologists', 'Memory', 'Developmental psychology'],
        'Economics': ['Economic theories', 'Nobel laureates in Economics', 'Financial markets', 'Cryptocurrencies', 'Economic indicators'],
        'Medicine': ['Human anatomy', 'Diseases', 'Medical procedures', 'Pharmaceuticals', 'Medical specialties'],
        'Sports': ['Olympic sports', 'World Cup winners', 'Tennis Grand Slams', 'NBA teams', 'Sports records'],
        'Technology': ['Tech companies', 'Programming frameworks', 'Internet protocols', 'Mobile devices', 'Cloud computing'],
        'Film': ['Academy Award winners', 'Film directors', 'Movie genres', 'Film studios', 'Cinematography'],
        'Politics': ['Political systems', 'International organizations', 'Treaties', 'World leaders', 'Elections'],
        'Astronomy': ['Planets', 'Constellations', 'Space missions', 'Astronomers', 'Galaxies'],
        'Environment': ['Climate change', 'Ecosystems', 'Endangered species', 'National parks', 'Renewable energy'],
        'Language': ['World languages', 'Linguistics', 'Writing systems', 'Etymology', 'Language families'],
        'Food': ['World cuisines', 'Cooking techniques', 'Spices', 'Wine regions', 'Traditional dishes'],
        'Architecture': ['Architectural styles', 'Famous buildings', 'Architects', 'Skyscrapers', 'Ancient structures'],
        'Law': ['Legal systems', 'Constitutional law', 'International law', 'Supreme Court cases', 'Legal terminology']
                }
        """
        print("\n📚 Strategy 3: Fetching from specific domain categories...")
        
        for domain, categories in domain_categories.items():
            print(f"\n🔍 Domain: {domain}")
            domain_articles = {}
            
            for category in categories:
                top_in_cat = self.get_category_top_viewed(category, limit=25, sample_size=100)
                for title, views in top_in_cat:
                    if title not in domain_articles or domain_articles[title] < views:
                        domain_articles[title] = views
            
            # Get top 100 for this domain
            sorted_domain = sorted(
                domain_articles.items(),
                key=lambda x: x[1],
                reverse=True
            )[:500]
            
            # Fetch content for top articles
            domain_results = []
            print(f"  Fetching content for top {len(sorted_domain)} {domain} articles...")
            
            for i, (title, views) in enumerate(sorted_domain[:500]):  # Limit to 50 for speed
                content = self.get_wikipedia_article_content(title)
                if content:
                    article_data = {
                        'rank': i + 1,
                        'title': title,
                        'category': domain,
                        'view_count': views,
                        'content': content,
                        'word_count': len(content.split())
                    }
                    domain_results.append(article_data)
                
                if (i + 1) % 10 == 0:
                    print(f"    Progress: {i+1}/{min(500, len(sorted_domain))}")
                
                # time.sleep(0.1)  # Rate limiting
            
            results[domain] = domain_results
            
            # Save domain results
            filename = f"wikipedia_smart_{domain.lower()}_top500.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(domain_results, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Saved {len(domain_results)} articles to '{filename}'")
        
        """
        # Handle famous people separately
        people_results = []
        print(f"\n👤 Fetching content for top {len(famous_people)} famous people...")
        
        for i, (title, views) in enumerate(famous_people[:500]):
            content = self.get_wikipedia_article_content(title)
            if content:
                article_data = {
                    'rank': i + 1,
                    'title': title,
                    'category': 'Famous People',
                    'view_count': views,
                    'content': content,
                    'word_count': len(content.split())
                }
                people_results.append(article_data)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{min(500, len(famous_people))}")
            
            # time.sleep(0.1)
        
        results['Famous People'] = people_results
        
        # Save famous people results
        filename = "wikipedia_smart_famous_people_top500.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(people_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(people_results)} famous people to '{filename}'")
        """
        return results

def main():
    """Main execution"""
    print("🚀 Smart Wikipedia Scraper")
    print("=" * 60)
    print("This optimized version uses multiple strategies to get high-quality")
    print("articles without scanning massive categories like 'Living people'")
    print("=" * 60)
    
    scraper = OptimizedWikipediaScraper()
    
    # Run the smart scraping
    start_time = time.time()
    results = scraper.smart_category_scrape()
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SCRAPING COMPLETE - SUMMARY")
    print("=" * 60)
    
    total_articles = sum(len(articles) for articles in results.values())
    print(f"✅ Total articles scraped: {total_articles}")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"⚡ Average time per article: {elapsed/total_articles:.1f} seconds")
    
    print("\n📁 Files created:")
    for domain in results:
        filename = f"wikipedia_smart_{domain.lower().replace(' ', '_')}_top500.json"
        print(f"  - {filename} ({len(results[domain])} articles)")

if __name__ == "__main__":
    main()