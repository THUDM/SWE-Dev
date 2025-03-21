import argparse
import json
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from urllib3.exceptions import MaxRetryError, ProxyError

GITHUB_API_URL = "https://api.github.com/search/repositories"
GITHUB_TOKENS = os.environ.get('GITHUB_TOKENS', '').split(',')
if not GITHUB_TOKENS or GITHUB_TOKENS[0] == '':
    raise ValueError("Please set GITHUB_TOKENS environment variable with comma-separated tokens")

def get_headers():
    token = random.choice(GITHUB_TOKENS)
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }

def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def fetch_repositories(query, page=1, per_page=100, max_retries=5):
    params = {
        "q": query,
        "page": page,
        "per_page": per_page,
        "sort": "stars",
        "order": "desc"
    }
    
    session = create_session()
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = session.get(
                GITHUB_API_URL, 
                headers=get_headers(), 
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:  # Rate limit
                print(f"Rate limit hit, waiting 60 seconds... (Attempt {retry_count + 1}/{max_retries})")
                time.sleep(60)
            elif response.status_code == 422:  # Only first 1000 results available
                print(f"422 Error (only 1000 results available), query: {query}")
                return {"items": []}
            else:
                print(f"Error {response.status_code}: {response.text}")
                time.sleep(10)
            
        except (ProxyError, MaxRetryError) as e:
            print(f"Proxy/Connection error: {e}")
            print("Waiting 30 seconds before retry...")
            time.sleep(30)
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            time.sleep(10)
        
        retry_count += 1
    
    return {"items": []}

def process_repo(repo):
    try:
        return {
            "github": repo["html_url"],
            "name": repo["name"],
            "owner": repo["owner"]["login"],
            "stars": repo["stargazers_count"],
            "description": repo.get("description", ""),
            "language": repo.get("language", ""),
            "created_at": repo.get("created_at", ""),
            "updated_at": repo.get("updated_at", "")
        }
    except Exception as e:
        print(f"Error processing repository: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='Scrape GitHub Python repositories')
    parser.add_argument('--max_repos', type=int, default=5000,
                       help='Maximum number of repositories to scrape')
    parser.add_argument('--output_file', type=str, default='python_repos.jsonl',
                       help='Output file for results')
    parser.add_argument('--workers', type=int, default=5,
                       help='Number of worker threads')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Delay between requests in seconds')
    return parser.parse_args()

def scrape_github_repositories(args):
    all_repositories = []
    page = 1
    consecutive_errors = 0
    max_consecutive_errors = 5

    # Define different query combinations
    star_ranges = [(100, 500), (501, 1000), (1001, 5000), (5000, 5000000)]
    years = range(2020, 2024)
    months = range(1, 13)
    topics = [
        "machine-learning", "data-science", "web", "django", "flask",
        "deep-learning", "nlp", "computer-vision", "pytorch", "tensorflow",
        "automation", "web-scraping", "robotframework", "scientific-computing",
        "cryptography", "cybersecurity", "game-development", "pygame",
        "blockchain", "cryptocurrency", "quantum-computing", "genomics",
        "bioinformatics", "sql", "pandas", "numpy", "matplotlib"
    ]
    
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for star_range in star_ranges:
            for year in years:
                for month in months:
                    for topic in topics:
                        try:
                            # Check termination conditions
                            if len(all_repositories) >= args.max_repos:
                                print(f"Reached maximum number of repositories: {args.max_repos}")
                                return all_repositories
                            
                            # Construct the query
                            date_range = f"{year}-{month:02d}-01..{year}-{month:02d}-28"
                            star_query = f"stars:{star_range[0]}..{star_range[1]}"
                            query = f"language:Python {star_query} created:{date_range} topic:{topic}"
                            
                            print(f"\nFetching repositories for query: {query}")
                            data = fetch_repositories(query, page=page, per_page=100)
                            repositories = data.get("items", [])
                            
                            if not repositories:
                                print("No more repositories found for this query")
                                continue

                            # Process repositories in parallel
                            futures = {executor.submit(process_repo, repo): repo for repo in repositories}
                            for future in as_completed(futures):
                                result = future.result()
                                if result:
                                    all_repositories.append(result)
                                    with open(args.output_file, "a") as f:
                                        f.write(json.dumps(result) + "\n")
                                    print(f"Processed repository: {result['github']}")

                            print(f"Completed page {page}, total repositories: {len(all_repositories)}")
                            consecutive_errors = 0
                            page += 1
                            
                            # Rate limiting delay
                            if args.delay > 0:
                                print(f"Waiting {args.delay} seconds before next page...")
                                time.sleep(args.delay)
                        
                        except KeyboardInterrupt:
                            print("\nReceived keyboard interrupt. Stopping gracefully...")
                            return all_repositories
                        except Exception as e:
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                print(f"Too many consecutive errors ({consecutive_errors}). Stopping.")
                                return all_repositories
                            print(f"Error fetching repositories: {e}")
                            traceback.print_exc()
                            time.sleep(10)

    return all_repositories

if __name__ == "__main__":
    args = parse_args()
    repositories = scrape_github_repositories(args)
    print(f"Scraping completed! Total repositories fetched: {len(repositories)}")