import argparse
import json
import re
import secrets
import string
import traceback
import xmlrpc.client
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import requests
from requests import HTTPError


def generate_random_string(length=10):
    alphabet = string.ascii_letters + string.digits 
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def fetch_all_packages(output_name):
    base_url = "https://pypi.org/pypi"
    xmlclient = xmlrpc.client.ServerProxy(base_url)
    print("Fetching package data from PyPI...")
    packages = xmlclient.list_packages_with_serial()
    with open(output_name, "w", encoding="utf-8") as file:
        json.dump(packages, file, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {output_name}.")
    print(f"Fetched {len(packages)} packages. Saved to 'pypi_packages.json'.")

base_url = "https://pypi.org/pypi"
session = requests.Session()

def user_agent_generator():
    return f"Pypi Daily Sync (Contact: {generate_random_string(10)}@gmail.com)"

def all_packages():
    xmlclient = xmlrpc.client.ServerProxy(base_url)
    return xmlclient.list_packages_with_serial()

def pkg_meta(name):
    resp = session.get(f"{base_url}/{name}/json", headers={'User-Agent': user_agent_generator()})
    resp.raise_for_status()
    return resp.json()

def extract_github_repo(url):
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
        except:
            traceback.print_exc()
            print("Warning! problems accessing pypi api. Will retry in 3s")
            sleep(3)
    urls = meta['info']['project_urls'].values()
    for url in urls:
        result = extract_github_repo(url).replace(".git", "")
        if result:
            print(f'get url: {result}, {output_file}')
            with open(output_file, 'a') as f:
                f.write(f'{result}\n')

def crawl_pkgs_meta(packages, output_file, workers):
    """
    Crawl metadata for a list of PyPI packages and extract GitHub repository URLs.
    """
    args_list = [(name, output_file) for name in packages]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(lambda args: save_pkg_meta(*args), args_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Path to pypi packages. You can get them using function `fetch_all_packages`")
    parser.add_argument("--output_name", type=str, help="Path to save urls")
    parser.add_argument("--workers", type=int, default=128, help="Concurrency count")
    args = parser.parse_args()
    with open(args.dataset_name, "r") as f:
        dataset = json.load(f)
    names = dataset.keys()
    crawl_pkgs_meta(names, args.output_name, workers=args.workers)

if __name__ == "__main__":
    main()