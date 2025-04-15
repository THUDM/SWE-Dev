import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
GITHUB_TOKENS = os.environ.get('GITHUB_TOKENS', '').split(',')
if not GITHUB_TOKENS or GITHUB_TOKENS[0] == '':
    raise ValueError("Please set the GITHUB_TOKENS environment variable with comma-separated tokens.")

def get_headers():
    token = GITHUB_TOKENS[0]
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }

# GraphQL Query Template for GitHub Repositories
def get_graphql_query(after_cursor=None, stars_range=">1000", repo_count=10): 
    after = f', after: "{after_cursor}"' if after_cursor else ""
    return f"""
    {{
      search(query: "stars:{stars_range} language:Python", type: REPOSITORY, first: {repo_count}{after}) {{
        repositoryCount
        pageInfo {{
          endCursor
          hasNextPage
        }}
        edges {{
          node {{
            ... on Repository {{
              name
              url
              stargazers {{
                totalCount
              }}
              forkCount
              owner {{
                login
              }}
            }}
          }}
        }}
      }}
    }}
    """
    
# Execute GraphQL query with retry logic for errors
def execute_query(query, retries=3, delay=10):
    for attempt in range(retries):
        response = requests.post(GITHUB_GRAPHQL_URL, json={"query": query}, headers=get_headers())
        if response.status_code == 200:
            json_response = response.json()
            if "data" not in json_response:
                print(f"Error: 'data' field is missing in the response. Full response: {json.dumps(json_response, indent=2)}")
                raise Exception("GraphQL API response does not contain 'data' field.")
            return json_response
        elif response.status_code in [502, 500, 504]:
            # Handle server errors by retrying after a delay
            print(f"GraphQL query failed with status code {response.status_code}: {response.text}")
            if attempt < retries - 1:
                print(f"Retrying query in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)  # Wait before retrying
            else:
                print(f"Failed after {retries} attempts.")
                raise Exception(f"GraphQL query failed with status code {response.status_code} after {retries} retries.")
        else:
            # For other errors, print details and raise the exception
            print(f"GraphQL query failed with status code {response.status_code}: {response.text}")
            response.raise_for_status()

# Process repository data
def process_repo(repo):
    return {
        "name": repo["node"]["name"],
        "url": repo["node"]["url"],
        "stars": repo["node"]["stargazers"]["totalCount"],
        "forks": repo["node"]["forkCount"],
        "description": repo["node"].get("description", ""),
        "language": repo["node"]["primaryLanguage"]["name"] if repo["node"].get("primaryLanguage") else None,
        "owner": repo["node"]["owner"]["login"]
    }

# Scrape GitHub repositories for a specific star range
def scrape_repositories_for_range(args, stars_range):
    all_repositories = []
    has_next_page = True
    cursor = None
    total_repos_fetched = 0
    repo_count_per_query = 10  # Fetch 10 repositories per query

    while has_next_page and total_repos_fetched < args.max_repos:
        query = get_graphql_query(after_cursor=cursor, stars_range=stars_range, repo_count=repo_count_per_query)
        print(f"Executing query for stars range {stars_range}, fetched {total_repos_fetched}/{args.max_repos} repositories.")
        
        try:
            result = execute_query(query)
            repos = result["data"]["search"]["edges"]
            # Process repository data
            for repo in repos:
                processed_repo = process_repo(repo)
                all_repositories.append(processed_repo)
                # Write to file
                with open(args.output_file, "a") as f:
                    f.write(json.dumps(processed_repo) + "\n")
                total_repos_fetched += 1

                if total_repos_fetched >= args.max_repos:
                    break
            
            # Get pagination info
            page_info = result["data"]["search"]["pageInfo"]
            has_next_page = page_info["hasNextPage"]
            cursor = page_info["endCursor"]

        except Exception as e:
            print(f"Error while fetching repositories for stars range {stars_range}: {e}")
            time.sleep(10)  # Wait before retrying
            break

        if args.delay > 0:
            time.sleep(args.delay)  # Delay between requests to avoid rate-limiting

    return all_repositories

# Generate star ranges in steps of 200
def generate_star_ranges(step=200, max_stars=200000):
    ranges = []
    for i in range(100, max_stars, step):
        start = i
        end = i + step - 1
        ranges.append(f"{start}..{end}")
    ranges.append(f">{max_stars}")  # For repositories with more than max_stars stars
    return ranges

# Scrape GitHub repositories using concurrent threads
def scrape_github_repositories(args):
    all_repositories = []
    star_ranges = generate_star_ranges(step=200)  # Generate star ranges in steps of 200

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_range = {executor.submit(scrape_repositories_for_range, args, stars_range): stars_range for stars_range in star_ranges}
        for future in as_completed(future_to_range):
            stars_range = future_to_range[future]
            try:
                repos = future.result()
                all_repositories.extend(repos)
                print(f"Finished fetching for stars range: {stars_range}")
            except Exception as e:
                print(f"Error in range {stars_range}: {e}")
                traceback.print_exc()

    return all_repositories

def parse_args():
    parser = argparse.ArgumentParser(description="Scrape GitHub repositories using GraphQL API.")
    parser.add_argument("--max_repos", type=int, default=1000, help="Maximum number of repositories to scrape.")
    parser.add_argument("--output_file", type=str, default="github_repos.jsonl", help="File to save the results in JSONL format.")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent threads.")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay in seconds between API requests.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.output_file, "w") as f:
        f.write("")  # Clear the output file
    repositories = scrape_github_repositories(args)
    print(f"Scraping completed! Fetched {len(repositories)} repositories.")