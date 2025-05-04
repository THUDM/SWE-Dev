import argparse
import os
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from threading import Lock

import jsonlines
from dotenv import load_dotenv
from swebench.collect.build_dataset import main as build_dataset
from swebench.collect.print_pulls import main as print_pulls
from tqdm import tqdm
from swedev.config import Config

load_dotenv()

downloaded_repos = set()
downloaded_repos_lock = Lock()

def split_instances(input_list: list, n: int) -> list:
    avg_length = len(input_list) // n
    remainder = len(input_list) % n
    result, start = [], 0

    for i in range(n):
        length = avg_length + 1 if i < remainder else avg_length
        sublist = input_list[start: start + length]
        result.append(sublist)
        start += length

    return result

def clone_repo(repo_name):
    global downloaded_repos
    base_dir = Config.local_repo_dir
    assert base_dir, "local_repo_dir not configured"
    
    repo = repo_name.split("/")[-1]
    
    with downloaded_repos_lock:
        if repo in downloaded_repos:
            print(f"Repository {repo} has already been cloned, skipping...")
            return

    repo_path = os.path.join(base_dir, repo)
    os.makedirs(repo_path, exist_ok=True)
    if os.path.exists(repo_path) and len(os.listdir(repo_path)):
        print(f"Repository {repo} already exists locally, skipping clone...")
        with downloaded_repos_lock:
            downloaded_repos.add(repo)
        return

    for retry in range(5):
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    f"https://github.com/{repo_name}.git",
                    repo_path
                ], 
                check=True,
                cwd=repo_path,
                shell=True
            )
            print(f"Successfully cloned {repo_name}")
            with downloaded_repos_lock:
                downloaded_repos.add(repo)
            break
        except Exception as e:
            print(f"Failed to clone {repo_name} (attempt {retry + 1}/5): {e}")
            time.sleep(2 ** retry)
    else:
        print(f"Failed to clone {repo_name} after 5 attempts.")

def process_repo(repo, output_folder, max_pulls, cutoff_date, token_iterator, do_clone=True):
    repo = repo.strip(",").strip()
    repo_name = repo.split("/")[1]

    token = next(token_iterator)
    try:
        path_prs = os.path.join(output_folder, "prs")
        path_tasks = os.path.join(output_folder, "tasks")
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(path_prs, exist_ok=True)
        os.makedirs(path_tasks, exist_ok=True)

        path_pr = os.path.join(path_prs, f"{repo_name}-prs.jsonl")
        if cutoff_date:
            path_pr = path_pr.replace(".jsonl", f"-{cutoff_date}.jsonl")
        if not os.path.exists(path_pr):
            print(f"Pull request data for {repo} not found, creating...")
            print_pulls(
                repo,
                path_pr,
                token,
                max_pulls=max_pulls,
                cutoff_date=cutoff_date
            )
            print(f"✅ Successfully saved PR data for {repo} to {path_pr}")
        else:
            print(f"📁 Pull request data for {repo} already exists at {path_pr}, skipping...")

        path_task = os.path.join(path_tasks, f"{repo_name}-task-instances.jsonl")
        if not os.path.exists(path_task):
            print(f"Task instance data for {repo} not found, creating...")
            build_dataset(path_pr, path_task, token)
            print(f"✅ Successfully saved task instance data for {repo} to {path_task}")
        else:
            print(f"📁 Task instance data for {repo} already exists at {path_task}, skipping...")
    except Exception as e:
        print("-" * 80)
        print(f"Something went wrong for {repo}, skipping: {e}")
        print("Here is the full traceback:")
        traceback.print_exc()
        print("-" * 80)

    if do_clone:
        clone_repo(repo)

def combine_results(output_folder: str,):
    print("Start combining results...")
    path_tasks = os.path.join(output_folder, "tasks")
    all_tasks = []
    files = os.listdir(path_tasks)
    for file in tqdm(files):
        file_path = os.path.join(path_tasks, file)
        if os.path.exists(file_path):
            with jsonlines.open(file_path, "r") as f:
                all_tasks.extend([d for d in f])
        else:
            print(f"Warning: {file_path} does not exist, skipping...")
    print("Writing!")
    with open(os.path.join(output_folder, "all_tasks.jsonl"), "w") as f:
        writer = jsonlines.Writer(f)
        writer.write_all(all_tasks)    
    print("Finished writing results.")

def main(
        repo_file: str,
        output_folder: str,
        max_pulls: int = None,
        cutoff_date: str = None,
        num_workers: int = 1,
        start_index: int = None,
        end_index: int = None,
        **kwargs
    ):

    with jsonlines.open(repo_file, "r") as f:
        repos = [d for d in f if d["github"]]
        repos = [d["github"]
                    .replace("http://github.com/", "")
                    .replace("https://github.com/", "")
                    .replace("git@github.com:", "")
                    .replace(".git", "")
                    .replace("github.com/", "")
                    for d in repos]
        for i in range(len(repos)):
            if repos[i].endswith("/"):
                repos[i] = repos[i][:-1]
            repos[i] = '/'.join(repos[i].split('/')[:2])
    
    print(f"Total repos: {len(repos)}")
    used = []
    used_path = f'{output_folder}/tasks'
    if os.path.exists(used_path):
         for file in os.listdir(used_path):
            if file.endswith("-instances.jsonl"):
                used.extend([file.replace("-task-instances.jsonl", "")])
    repos = [r for r in repos if not r.split("/")[-1] in used]
    print(f"Remaining repos: {len(repos)}")
    if start_index is not None or end_index is not None:
        repos = repos[start_index:end_index]
    repos = reversed(repos)
    tokens = Config.github_tokens
    if not tokens: 
        raise Exception("Missing github_tokens in configuration, add to config file or set GITHUB_TOKENS environment variable")
    tokens = [t.strip() for t in tokens.split(",")]
    token_iterator = cycle(tokens)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_repo, repo, output_folder, max_pulls, cutoff_date, token_iterator)
            for repo in repos
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing repository: {str(e)}")

    path_tasks = os.path.join(output_folder, "tasks")
    all_tasks = []
    files = os.listdir(path_tasks)
    for file in files:
        file_path = os.path.join(path_tasks, file)
        if os.path.exists(file_path):
            with jsonlines.open(file_path, "r") as f:
                all_tasks.extend([d for d in f])
        else:
            print(f"Warning: {file_path} does not exist, skipping...")

    with open(os.path.join(output_folder, "all_tasks.jsonl"), "w") as f:
        writer = jsonlines.Writer(f)
        writer.write_all(all_tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub Repo Data Collection")
    parser.add_argument("--repo_file", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--max_pulls", type=int, help="Maximum number of pulls to log", default=None)
    parser.add_argument("--cutoff_date", type=str, help="Cutoff date for PRs to consider in format YYYYMMDD", default=None)
    parser.add_argument("--num_workers", type=int, help="Parallel worker count.")
    parser.add_argument("--start_index", type=int, help="Start index of the repository list", default=None)
    parser.add_argument("--end_index", type=int, help="End index of the repository list", default=None)
    parser.add_argument("--combine_results", action="store_true")
    parser.add_argument("--do_clone", action="store_true")
    args = parser.parse_args()
    if args.combine_results:
        combine_results(args.output_folder)
        exit()
    main(**vars(args))