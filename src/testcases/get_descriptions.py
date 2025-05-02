import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List

import jsonlines
from src.utils.error_handler import *
from src.utils.extract_signs import *
from src.utils.prompts import *
from src.utils.utils import *
from tqdm import tqdm
from src.config import CONDA_BASE, PLAYGROUND_PATH, OPENAI_BASE_MODEL, OPENAI_BASE_URL, get_config_value

call_counter = tqdm(desc="API Calls", unit="calls")
total_counter = tqdm(desc="Progress", unit="items")
saved_counter = tqdm(desc="Saved", unit="items")
start_time = time.time()

DEBUG = False
REVISE_ROUNDS = 2

def test_formatter(testcase):
    return TESTCASE_FORMAT.format(testcase["content"], testcase["env"])

def setup_logging(output_folder, console=False):
    log_dir = os.path.join(output_folder, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'test_iter_generation_{timestamp}.log')
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    handlers = []
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logger

def parse_testcase(text):
    commands = [
        'rm', 'git clone', 'dd', 'mkfs', 'shutdown', 'reboot', 'kill', 'mv', 'scp', 'ifconfig',
        'rsync', 'pkill', 'docker rm', 'docker rmi', 'iptables', 'ufw', 'mount', 'umount',
    ]    
    install_commands = ['pip install', 'apt-get', 'apt']
    
    def extract_content(tag, text):
        """Helper function to extract content inside specific tags."""
        pattern = fr"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    def extract_code_blocks(content):
        """Helper function to extract code blocks enclosed by triple backticks."""
        pattern = r"```(?:[\w]*)\n(.*?)```"
        return re.findall(pattern, content, re.DOTALL)    
    
    def extract_imports(code):
        """Helper function to extract import statements from code."""
        pattern = r"^(?:import\s+\w+|from\s+\w+\s+import\s+\w+)"
        imports = re.findall(pattern, code, re.MULTILINE)
        return imports

    def extract_imports(code):
        """Helper function to extract import statements from code."""
        pattern = r"^(?:import\s+([\w.]+)|from\s+([\w.]+)\s+import\s+\w+)"
        matches = re.findall(pattern, code, re.MULTILINE)
        packages = {match.split('.')[0] for pair in matches for match in pair if match}
        return packages

    def generate_pip_commands(imports):
        """Generate pip install commands from import statements."""
        return [f"pip install {pkg}" for pkg in imports]
    
    def replace_os_system(code):
        """Replace os.system calls with pass."""
        lines = code.splitlines()
        result = []
        
        for line in lines:
            if "os.system" in line and any([command in line for command in commands]):
                indent = len(line) - len(line.lstrip())
                result.append(" " * indent + "pass")
            else:
                result.append(line)
                
        return "\n".join(result)
    
    testcases_raw = extract_content("testcase", text)
    envs_raw = extract_content("env", text)
    testcases = [extract_code_blocks(tc) for tc in testcases_raw]
    envs = [extract_code_blocks(env) for env in envs_raw]

    testcases = [block for blocks in testcases for block in blocks]
    envs = [block for blocks in envs for block in blocks]

    testcases = [replace_os_system(tc) for tc in testcases]

    if len(testcases) > len(envs):
        envs.extend([""] * (len(testcases) - len(envs)))
    elif len(envs) > len(testcases):
        envs = envs[:len(testcases)]

    for i in range(len(envs)):
        ret = ''
        for line in envs[i].splitlines():
            if any(element in line for element in install_commands):
                ret += line + '\n'
        ret = ret.strip()

        imports = extract_imports(testcases[i])
        pip_commands = generate_pip_commands(imports)
        if pip_commands:
            ret = '\n'.join(pip_commands) + '\n' + ret
        envs[i] = ret

    return envs, testcases

def process_single_instance(loc: Dict, args: argparse.Namespace, logger: logging.Logger) -> Dict:
    """Process a single test instance"""
    instance_id = loc["instance_id"]
    repo = loc["repo"]
    statement = loc["problem_statement"]
    patch = loc["patch"]
    commit_id = loc["base_commit"]
    repo_id = f'{instance_id}_{repo.replace("/", "_")}_{commit_id}'
    repo_playground = os.path.join(PLAYGROUND_PATH, repo_id, repo.split("/")[-1])

    try:
        statement = loc['problem_statement']
        repo = loc['repo']
        patch = loc['patch']
        hints_text = loc['hints_text']

        def tracked_call(*args, **kwargs):
            call_counter.update(1)
            elapsed = time.time() - start_time
            current_rps = call_counter.n / elapsed if elapsed > 0 else 0
            call_counter.set_postfix({'RPS': f'{current_rps:.2f}'})
            return call(*args, **kwargs)

        description_model = get_config_value("description.model", OPENAI_BASE_MODEL)
        description_base_url = get_config_value("description.base_url", OPENAI_BASE_URL)

        raw_desc, _ = tracked_call(
            messages=[{"role": "user", "content": SUMMARIZE_GHERKIN_TEST.format(repo, statement, patch, hints_text)}],
            max_tokens=2048,
            model=description_model,
            base_url=description_base_url,
            logger=logger                             
        )
        if raw_desc == "Error":
            print("Too long when generating raw desc")
            raw_desc, _ = tracked_call(
                messages=[{"role": "user", "content": SUMMARIZE_GHERKIN_TEST.format(repo, statement, patch, "No Hints Text Provided")}],
                max_tokens=2048,
                model=description_model,
                base_url=description_base_url,
                logger=logger
            )     
        
        desc, _ = tracked_call(
            messages=[{"role": "user", "content": MAKE_GHERKIN_TEST.format(repo, statement, patch, hints_text, raw_desc)}],
            max_tokens=2048,
            model=description_model,
            base_url=description_base_url,
            logger=logger
        )
        if desc == "Error":
            print("Too long when generating desc")
            desc, _ = tracked_call(
                messages=[{"role": "user", "content": SUMMARIZE_GHERKIN_TEST.format(repo, statement, patch, "No Hints Text Provided")}],
                max_tokens=2048,
                model=description_model,
                base_url=description_base_url,
                logger=logger
            )
        pattern = r'```(?:gherkin\n|\n)(.*?)\n```'
        descs = re.findall(pattern, desc, re.DOTALL)

        return {
            "repo": repo,
            "instance_id": instance_id,
            "problem_statement": statement,
            "patch": patch,
            "created_at": loc["created_at"],
            "hints_text": loc["hints_text"],
            "base_commit": loc["base_commit"],
            "descs": descs,
            "model": description_model
        }
        
    except Exception as e:
        logging.info(e)
        os.system(f"rm -rf {repo_playground}")
        os.system(f"rm -rf {CONDA_BASE}/envs/{instance_id}")
        # logger.error(f"Error processing instance {instance_id}: {str(e)}")
        # traceback.print_exc()
        return {
            "repo": repo,
            "instance_id": instance_id,
            "problem_statement": statement,
            "patch": patch,
            "created_at": loc["created_at"],
            "hints_text": loc["hints_text"],
            "base_commit": loc["base_commit"],
            "descs": [],
            "model": get_config_value("description.model", OPENAI_BASE_MODEL)
        }
        
def make_testcases(args, logger: logging.Logger):
    logging.basicConfig(
        filename=f"{args.output_folder}/make_testcases.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.loc_file, 'r', encoding='latin-1') as f:
        locs = [json.loads(d) for d in f.readlines()]    

    print(f"Total {len(locs)} instances")
    total_counter.total = len(locs)
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='latin-1') as f:
            prev_o = [json.loads(d) for d in f.readlines()]
        prev_o_ids = [o["instance_id"] for o in prev_o]
        saved_counter.n = len(prev_o)
        saved_counter.refresh()
    else:
        prev_o_ids = []
        
    locs = [loc for loc in locs if loc["instance_id"] not in prev_o_ids]
    print(f"Remaining {len(locs)} instances")
    results = []
    result_lock = threading.Lock()

    def process_and_save(loc, output_file, logger):
        result = process_single_instance(loc, args, logger)
        total_counter.update(1)
        if result:
            with result_lock:
                if result["descs"]:
                    results.append(result)
                    with open(output_file, "a") as f:
                        f.write(f'{json.dumps(result)}\n')
                    saved_counter.update(1)
                else:
                    print(f"Skipping instance {loc['instance_id']} due to no descriptions")
                    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(lambda loc: process_and_save(loc, args.output_file, logger), locs), total=len(locs)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_file", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument(
        "--select_id",
        type=int,
        default=-1,
        help="Index the selected samples during post-processing.",
    )
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    logger = setup_logging(args.output_folder, console=True)
    args.output_file = os.path.join(args.output_folder, "output.jsonl")
    make_testcases(args, logger)

if __name__ == "__main__":
    main()
