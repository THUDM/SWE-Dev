import argparse
import json
import multiprocessing as mp
import os
import random
from multiprocessing import Lock, Manager, Pool

from bs4 import BeautifulSoup
from ghapi.core import GhApi
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tqdm import tqdm
from src.config import GITHUB_TOKENS

if not GITHUB_TOKENS:
    msg = "GitHub tokens not configured. Please configure GITHUB_TOKENS in your config file or set the environment variable."
    raise ValueError(msg)
apis = [GhApi(token=gh_token) for gh_token in GITHUB_TOKENS]
print("GH_tokens:", GITHUB_TOKENS)
def get_api():
    return random.sample(apis, 1)[0]

def setup_driver():
    """Setup and return a Chrome webdriver"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    return webdriver.Chrome(options=options)

def process_package(args):
    """Process a single package"""
    idx, title, href = args
    driver = setup_driver()
    
    try:
        package_name = title
        package_url = href

        # Get github URL
        package_github = None
        driver.get(package_url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for link in soup.find_all("a", class_="vertical-tabs__tab--with-icon"):
            found = False
            for x in ["Source", "Code", "Homepage"]:
                if (
                    x.lower() in link.get_text().lower()
                    and "github" in link["href"].lower()
                ):
                    package_github = link["href"]
                    found = True
                    break
            if found:
                break

        # Get stars and pulls from github API
        stars_count, pulls_count = None, None
        if package_github is not None:
            repo_parts = package_github.split("/")[-2:]
            owner, name = repo_parts[0], repo_parts[1]

            try:
                repo = get_api().repos.get(owner, name)
                stars_count = int(repo["stargazers_count"])
                issues = get_api().issues.list_for_repo(owner, name)
                pulls_count = int(issues[0]["number"])
            except:
                pass

        result = {
            "rank": idx,
            "name": package_name,
            "url": package_url,
            "github": package_github,
            "stars": stars_count,
            "pulls": pulls_count,
        }
        
        return result

    except Exception as e:
        print(f"Error processing package {package_name}: {str(e)}")
        return None
    
    finally:
        driver.quit()

def get_package_stats(data_tasks, output_file, num_workers, start_at=0):
    """
    Get package stats from PyPI page using multiple processes

    Args:
        data_tasks (list): List of packages + HTML
        output_file (str): File to write to
        num_workers (int): Number of worker processes
        start_at (int): Index to start processing from
    """
    processed_urls = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_urls.add(data["url"])
                except:
                    continue

    tasks = [
        (idx, chunk["title"], chunk["href"]) 
        for idx, chunk in enumerate(data_tasks[start_at:], start_at)  # Start from start_at index
        if chunk["href"] not in processed_urls
    ]

    if not tasks:
        print("All packages have been processed already")
        return

    with Pool(processes=num_workers) as pool:
        results = []
        for result in tqdm(
            pool.imap_unordered(process_package, tasks),
            total=len(tasks),
            desc="Processing packages"
        ):
            if result:
                results.append(result)
        
        with open(output_file, "a") as f:
            for res in results:
                print(json.dumps(res), file=f, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_repos", help="Maximum number of repos to get", type=int, default=5000)
    parser.add_argument("--output_folder", type=str, default="results/issues")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--start_at", type=int, default=0, help="Index to start processing packages from")
    args = parser.parse_args()

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    
    print(f"Now getting top {args.max_repos} pypi packages")
    url_top_pypi = "https://hugovk.github.io/top-pypi-packages/"
    driver = webdriver.Chrome(options=options)
    
    try:
        print("Chrome start successfully!")
        driver.get(url_top_pypi)
        button = driver.find_element(By.CSS_SELECTOR, 'button[ng-click="show(8000)"]')
        button.click()

        print("Getting package stats")
        soup = BeautifulSoup(driver.page_source, "html.parser")
        package_list = soup.find("div", {"class": "list"})
        packages = package_list.find_all("a", class_="ng-scope")
        
        package_data = [
            {"title": pkg.get_text(), "href": pkg["href"]} 
            for pkg in packages
        ]
        print(f"Will save to {args.output_folder}")
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
            
        output_file = f"{args.output_folder}/pypi_rankings.jsonl"
        get_package_stats(
            package_data, 
            output_file,
            args.num_workers,
            start_at=args.start_at  # Pass the start_at argument here
        )
        
    finally:
        driver.quit()

if __name__ == "__main__":
    main()