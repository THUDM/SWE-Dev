import argparse
import json
import os
import random
import time
from multiprocessing import Pool

from bs4 import BeautifulSoup
from ghapi.core import GhApi
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tqdm import tqdm
from swedev.config import Config

if not Config.github_tokens:
    msg = "GitHub tokens not configured. Please configure github_tokens in your config file or set the GITHUB_TOKENS environment variable."
    raise ValueError(msg)
apis = [GhApi(token=gh_token) for gh_token in Config.github_tokens]
print("GitHub tokens:", Config.github_tokens)

def get_api():
    return random.choice(apis)

def setup_driver():
    """Setup and return a Chrome webdriver"""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--enable-javascript')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36')
    return webdriver.Chrome(options=options)

def process_package(args):
    """Process a single package"""
    idx, title, href = args
    driver = setup_driver()
    
    try:
        package_name = title
        package_url = href

        package_github = None
        driver.get(package_url)
        time.sleep(2)  # Wait for the page to load
        
        try:
            # Try to find GitHub link using JavaScript
            github_link = driver.execute_script("""
                const links = Array.from(document.querySelectorAll('a.vertical-tabs__tab--with-icon'));
                for (const link of links) {
                    const text = link.textContent.toLowerCase();
                    const href = link.href.toLowerCase();
                    if ((text.includes('source') || text.includes('code') || text.includes('homepage')) && href.includes('github')) {
                        return link.href;
                    }
                }
                return null;
            """)
            
            if github_link:
                package_github = github_link
        except:
            # Fallback to BeautifulSoup if JavaScript execution fails
            soup = BeautifulSoup(driver.page_source, "html.parser")
            for link in soup.find_all("a", class_="vertical-tabs__tab--with-icon"):
                found = False
                for x in ["source", "code", "homepage"]:
                    if (
                        x in link.get_text().lower()
                        and "github" in link["href"].lower()
                    ):
                        package_github = link["href"]
                        found = True
                        break
                if found:
                    break

        stars_count, pulls_count = None, None
        if package_github is not None:
            try:
                # Extract owner and repo name
                if "github.com" in package_github:
                    repo_parts = package_github.split("github.com/")[-1].split("/")
                    if len(repo_parts) >= 2:
                        owner, name = repo_parts[0], repo_parts[1].split("#")[0].split("?")[0]
                        
                        repo = get_api().repos.get(owner, name)
                        stars_count = int(repo["stargazers_count"])
                        pulls = get_api().pulls.list(owner, name, state="all", per_page=1)
                        if pulls:
                            pulls_count = pulls[0]["number"]
            except Exception as e:
                print(f"Error getting GitHub stats for {package_name}: {str(e)}")

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
        print(f"Error processing package {title}: {str(e)}")
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
    print(f"Processing {len(data_tasks)} packages")
    
    processed_urls = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_urls.add(data["url"])
                except:
                    continue

    tasks = [
        (idx, chunk["title"], chunk["href"]) 
        for idx, chunk in enumerate(data_tasks[start_at:], start_at)
        if chunk["href"] not in processed_urls
    ]

    if not tasks:
        print("All packages have been processed already")
        return

    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_package, tasks),
            total=len(tasks),
            desc="Processing packages"
        ):
            if result:
                with open(output_file, "a") as f:
                    print(json.dumps(result), file=f, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_repos", help="Maximum number of repos to get", type=int, default=5000)
    parser.add_argument("--output_folder", type=str, default="results/packages")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--start_at", type=int, default=0, help="Index to start processing packages from")
    args = parser.parse_args()

    url_top_pypi = "https://hugovk.github.io/top-pypi-packages/"
    driver = setup_driver()
    
    try:
        print("Chrome started successfully!")
        driver.get(url_top_pypi)
        
        # Wait for page to fully load
        time.sleep(5)
        
        try:
            # Use JavaScript to click the button
            driver.execute_script("""
                const buttons = Array.from(document.querySelectorAll('button'));
                for (const button of buttons) {
                    if (button.textContent.includes('15000')) {
                        button.click();
                        return true;
                    }
                }
                return false;
            """)
            print("Clicked button via JavaScript")
        except:
            # Fallback to selenium if JavaScript fails
            try:
                button = driver.find_element(By.CSS_SELECTOR, 'button[ng-click="show(15000)"]')
                button.click()
                print("Clicked button via Selenium")
            except:
                print("Failed to click button, trying to find other versions")
                buttons = driver.find_elements(By.TAG_NAME, 'button')
                for btn in buttons:
                    if "15000" in btn.text:
                        btn.click()
                        print("Found and clicked alternative button")
                        break
        
        # Wait for the content to load (longer wait time)
        time.sleep(10)

        print("Getting package stats")
        
        package_data = driver.execute_script("""
            const packages = Array.from(document.querySelectorAll('div.list a.ng-scope'));
            return packages.map(pkg => {
                const fullText = pkg.textContent.trim();
                // Extract just the package name, removing rank and download numbers
                const packageName = fullText.split('\\n')[1].trim();
                return {
                    title: packageName,
                    href: pkg.href
                };
            });
        """)
        
        if not package_data:
            print("JavaScript extraction failed, using BeautifulSoup...")
            soup = BeautifulSoup(driver.page_source, "html.parser")
            package_list = soup.find("div", {"class": "list"})
            
            if not package_list:
                print("BeautifulSoup couldn't find package list, using WebDriver directly...")
                packages = driver.find_elements(By.CSS_SELECTOR, 'div.list a.ng-scope')
                package_data = []
                for pkg in packages:
                    full_text = pkg.text.strip()
                    # Extract just the package name, removing rank and download numbers
                    parts = full_text.split('\n')
                    if len(parts) > 1:
                        package_name = parts[1].strip()
                        package_data.append({"title": package_name, "href": pkg.get_attribute("href")})
            else:
                packages = package_list.find_all("a", class_="ng-scope")
                package_data = []
                for pkg in packages:
                    full_text = pkg.get_text().strip()
                    # Extract just the package name, removing rank and download numbers
                    parts = full_text.split('\n')
                    if len(parts) > 1:
                        package_name = parts[1].strip() 
                        package_data.append({"title": package_name, "href": pkg["href"]})
                        
        print(f"Found {len(package_data)} packages, will use top {args.max_repos} packages!")
        
        package_data = package_data[:args.max_repos]
        
        print(f"Will save to {args.output_folder}")
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
            
        output_file = f"{args.output_folder}/pypi_rankings.jsonl"
        get_package_stats(
            package_data, 
            output_file,
            args.num_workers,
            start_at=args.start_at
        )

    finally:
        driver.quit()

if __name__ == "__main__":
    main()