import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from math import ceil

import requests
import aiohttp
import asyncio
import pandas as pd
from tqdm import tqdm
from src.config import Config

# Use GitHub tokens from config
if not Config.github_tokens:
    raise ValueError("GitHub tokens not configured. Please configure github_tokens in your config file or set the GITHUB_TOKENS environment variable.")

def get_token():
    """Randomly select a GitHub token from the list."""
    return random.choice(Config.github_tokens)

def fetch_github_data(url, headers, max_retries=12):
    """
    Fetch stars and pull request counts from a single GitHub repository URL.

    :param url: GitHub repository URL.
    :param headers: Headers for the GitHub API request.
    :param max_retries: Maximum number of retries for failed requests.
    :return: A dictionary with the repository URL, stars, and pull request count, or None if an error occurs.
    """
    retries = 0
    while retries <= max_retries:
        try:
            # Extract owner and repo name from the GitHub URL
            parts = url.rstrip("/").split("/")
            if len(parts) < 5 or parts[2] != "github.com":
                print(f"Invalid GitHub URL skipped: {url}")
                return None
            owner, repo = parts[-2], parts[-1]

            # Base API URL for the repository
            base_api_url = f"https://api.github.com/repos/{owner}/{repo}"

            # Fetch repository details (stars)
            headers["Authorization"] = f"Bearer {get_token()}"
            repo_response = requests.get(base_api_url, headers=headers)
            if repo_response.status_code != 200:
                print(
                    f"Failed to fetch repo data for {url}: {repo_response.status_code}, {repo_response.text}")
                raise requests.exceptions.RequestException("Repo request failed")
            repo_data = repo_response.json()
            stars = repo_data.get("stargazers_count", 0)

            pulls_url = f"{base_api_url}/pulls"
            headers["Authorization"] = f"Bearer {get_token()}"
            pulls_response = requests.get(
                pulls_url, headers=headers, params={"state": "all", "per_page": 1})
            if pulls_response.status_code != 200:
                print(
                    f"Failed to fetch pull request data for {url}: {pulls_response.status_code}")
                raise requests.exceptions.RequestException("Pull request failed")
            pulls_count = 0
            if "Link" in pulls_response.headers:
                links = _parse_link_header(pulls_response.headers["Link"])
                if "last" in links:
                    last_url = links["last"]
                    pulls_count = int(last_url.split("page=")[-1])
            else:
                pulls_count = len(pulls_response.json())
            return {"github": url, "stars": stars, "pulls": pulls_count}

        except Exception as e:
            if repo_response.status_code == 404:
                return None
            
            retries += 1
            if retries > max_retries:
                print(f"Error processing {url} after {max_retries} retries: {e}")
                return None
            wait_time = 2 ** retries
            print(f"Retrying {url} in {wait_time} seconds... (Attempt {retries})")
            time.sleep(wait_time)


def _parse_link_header(link_header):
    """
    Parse the GitHub API Link header for pagination URLs.

    :param link_header: The Link header string.
    :return: A dictionary with keys like 'next', 'prev', 'last', etc.
    """
    links = {}
    for part in link_header.split(","):
        section = part.split(";")
        if len(section) < 2:
            continue
        url = section[0].strip().strip("<>")
        rel = section[1].strip().split("=")[1].strip('"')
        links[rel] = url
    return links


def load_processed_urls(output_file, not_found_file="not_found_repos.txt"):
    """
    Load already processed URLs from the output file and the not_found_repos.txt file to avoid duplicate processing.

    :param output_file: Path to the output .jsonl file.
    :param not_found_file: Path to the not_found_repos.txt file.
    :return: A set of already processed URLs.
    """
    processed_urls = set()

    # Load URLs from the output file
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            processed_urls.update(json.loads(line)["github"] for line in f)

    # Load URLs from the not_found_repos.txt file
    if os.path.exists(not_found_file):
        with open(not_found_file, "r") as f:
            processed_urls.update(line.strip() for line in f if line.strip())

    return processed_urls


def process_urls(input_file, output_file, max_workers):
    """
    Process all GitHub URLs from the input file and save results to the output file.

    :param input_file: Path to the input .txt file containing GitHub URLs.
    :param output_file: Path to the output .jsonl file for saving results.
    :param max_workers: Number of threads for concurrent processing.
    """
    with open(input_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    processed_urls = load_processed_urls(output_file, "not_found_repos.txt")
    urls_to_process = [url for url in urls if url not in processed_urls]

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {get_token()}"
    }

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(urls_to_process)) as progress:
        future_to_url = {executor.submit(fetch_github_data, url, headers, 5): url for url in urls_to_process}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                if data:
                    with open(output_file, "a") as f:
                        f.write(json.dumps(data) + "\n")
                    results.append(data)
                    progress.update(1)
                    print(f"Processed: {url} -> Stars: {data['stars']}, Pulls: {data['pulls']}")
            except Exception as e:
                print(f"Error processing {url}: {e}")

def main():
    """
    Main function to parse arguments and execute the script.
    """
    parser = argparse.ArgumentParser(description="Fetch stars and pull request counts for GitHub repositories.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .txt file containing GitHub URLs.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .jsonl file for saving results.")
    parser.add_argument("--workers", type=int, default=10, help="Number of threads for concurrent processing.")
    args = parser.parse_args()

    process_urls(args.input_file, args.output_file, args.workers)


if __name__ == "__main__":
    main()