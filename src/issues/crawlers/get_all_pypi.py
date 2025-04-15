import asyncio
import json
import logging
import os
import random
import traceback
from datetime import datetime
from itertools import cycle

import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm

os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(f'logs/pypi_scan_{datetime.now():%Y%m%d_%H%M%S}.log'))

result_file = f'results/valid_packages_{datetime.now():%Y%m%d_%H%M%S}.jsonl'

async def get_all_packages():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://pypi.org/simple/') as response:
            text = await response.text()
            return [line.split('"')[1] for line in text.split('\n') if 'href' in line]

async def check_package(session, package, token, pbar):
    try:
        package = package.replace("/simple/", "")
        package = "project/" + package
        async with session.get(f'https://pypi.org/{package}') as response:
            if response.status != 200:
                return None
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')

            github_links = soup.select('a[href*="github.com"]')
            github_url = None
            
            for link in github_links:
                href = link.get('href', '')
                if 'github.com' in href:
                    github_url = href.split('#')[0]
                    if github_url.endswith('/'):
                        github_url = github_url[:-1].replace("issues", "")
                    if github_url.endswith(".git"):
                        github_url = github_url[:-4]
                    break
            
            if not github_url:
                return None

            logger.info(f"Found GitHub URL for {package}: {github_url}")
            github_api_url = github_url.replace('github.com', 'api.github.com/repos')
            headers = {'Authorization': f'token {token}'}
            
            while True:
                async with session.get(github_api_url, headers=headers) as repo_response:
                    if repo_response.status == 403:
                        logger.warning(f"Rate limit hit. Retrying in 5 minutes...")
                        await asyncio.sleep(300)  # Sleep for 5 minutes before retrying
                        continue
                    elif repo_response.status != 200:
                        return None
                    
                    repo_data = await repo_response.json()
                    stars = repo_data.get('stargazers_count', -1)
                    pulls = repo_data.get('open_issues_count', -1)
                    break  # Exit loop if successful

            result = {
                'name': package,
                'github': github_url,
                'stars': stars,
                'pulls': pulls,
                'timestamp': datetime.now().isoformat()
            }
            # Immediately write to file
            with open(result_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
            return result
                
    except Exception as e:
        logger.error(f"Error processing {package}: {str(e)}")
        traceback.print_exc()
    finally:
        pbar.update(1)
    
    return None

async def main():
    github_tokens = os.getenv('GITHUB_TOKENS', '').split(',')
    if not github_tokens or not github_tokens[0]:
        raise ValueError("GITHUB_TOKENS environment variable is required")
    token_cycle = cycle(github_tokens)

    logger.info("Fetching package list...")
    packages = await get_all_packages()
    random.shuffle(packages)
    chunk_size = 50

    async with aiohttp.ClientSession() as session:
        with tqdm(total=len(packages)) as pbar:
            for i in range(0, len(packages), chunk_size):
                chunk = packages[i:i + chunk_size]
                tasks = []
                
                for package in chunk:
                    token = next(token_cycle)
                    tasks.append(
                        asyncio.create_task(
                            check_package(session, package, token, pbar)
                        )
                    )
                
                await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())