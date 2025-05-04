import argparse
import json
import re
import secrets
import string
import xmlrpc.client
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import requests
from requests import HTTPError


def generate_random_string(length=10):
    alphabet = string.ascii_letters + string.digits 
    return ''.join(secrets.choice(alphabet) for _ in range(length))

base_url = "https://pypi.org/pypi"
session = requests.Session()

def user_agent_generator():
    return f"Pypi Daily Sync (Contact: {generate_random_string(10)}@gmail.com)"

def all_packages():
    xmlclient = xmlrpc.client.ServerProxy(base_url)
    print("Fetching all package names from PyPI...")
    return xmlclient.list_packages_with_serial()

def pkg_meta(name):
    resp = session.get(f"{base_url}/{name}/json", headers={'User-Agent': user_agent_generator()})
    resp.raise_for_status()
    return resp.json()

def extract_github_repo(url):
    if not url:
        return None
    pattern = r'^(https?:\/\/github\.com\/([a-zA-Z0-9._-]+)\/([a-zA-Z0-9._-]+))(\/.*)?$'
    match = re.match(pattern, url)
    if match:
        return match.group(1)  
    return None

def save_pkg_meta(name, output_file):
    api_success = False
    while not api_success:
        try:
            meta = pkg_meta(name)
            api_success = True
        except HTTPError as e:
            if e.response.status_code == 404:
                return
            print(f"HTTP error {e.response.status_code} for package {name}. Retrying in 3s...")
            sleep(3)
        except Exception as e:
            print(f"Error with package {name}: {str(e)}. Retrying in 3s...")
            sleep(3)
    
    try:
        project_urls = meta['info'].get('project_urls', {}) or {}
        if isinstance(project_urls, dict):
            urls = project_urls.values()
        else:
            urls = []
            
        homepage = meta['info'].get('home_page')
        if homepage:
            urls = list(urls) + [homepage]
            
        if meta['info'].get('package_url'):
            urls = list(urls) + [meta['info'].get('package_url')]
            
        for url in urls:
            github_url = extract_github_repo(url)
            if github_url:
                github_url = github_url.replace(".git", "")
                print(f'Found GitHub URL: {github_url} for package {name}')
                
                entry = {
                    "package_name": name,
                    "github": github_url
                }
                
                with open(output_file, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
                
                break
    except Exception as e:
        print(f"Error processing metadata for {name}: {str(e)}")

def crawl_github_urls(output_file, workers=128):
    """
    Get all PyPI packages and extract GitHub repository URLs in one step.
    """
    packages = all_packages()
    print(f"Found {len(packages)} packages. Starting GitHub URL extraction...")
    
    open(output_file, 'w').close()
    
    package_names = list(packages.keys())
    args_list = [(name, output_file) for name in package_names]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(lambda args: save_pkg_meta(*args), args_list)

def main():
    parser = argparse.ArgumentParser(description="Extract GitHub URLs for PyPI packages")
    parser.add_argument("--output", type=str, default="github_urls.jsonl", 
                        help="Path to save GitHub URLs in JSONL format (default: github_urls.jsonl)")
    parser.add_argument("--workers", type=int, default=128, 
                        help="Number of concurrent workers (default: 128)")
    args = parser.parse_args()
    
    crawl_github_urls(args.output, args.workers)
    print(f"Finished! GitHub URLs saved to {args.output} in JSONL format")

if __name__ == "__main__":
    main()