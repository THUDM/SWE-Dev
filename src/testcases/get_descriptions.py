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
from typing import Dict, List, Any, Set, Tuple
from pathlib import Path

import jsonlines
from src.utils.error_handler import *
from src.utils.extract_signs import *
from src.utils.prompts import *
from src.utils.utils import *
from tqdm import tqdm
from src.config import Config

call_counter = tqdm(desc="API Calls", unit="calls")
total_counter = tqdm(desc="Progress", unit="items")
saved_counter = tqdm(desc="Saved", unit="items")
start_time = time.time()

# Use Config.Testcase for settings
DEBUG = Config.Testcase.debug
REVISE_ROUNDS = Config.Testcase.revise_rounds

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
    repo_playground = os.path.join(Config.playground_path, repo_id, repo.split("/")[-1])

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

        description_model = Config.Description.model
        description_base_url = Config.Description.base_url

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
        os.system(f"rm -rf {Config.conda_base}/envs/{instance_id}")
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
            "model": Config.Description.model
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

def make_conda_playground(cmd_or_str: str, folder_name: str = "", log_only=True, copy_files=True) -> str:
    """
    Creates a conda playground environment, runs a given command or string, and returns the output
    """
    conda_base = Config.conda_base
    playground_path = Config.playground_path

    sub_process = True
    if not folder_name:
        folder_name = str(int(time.time()))
    if not os.path.exists(conda_base):
        print(f"WARNING: Conda base directory {conda_base} does not exist. Falling back to running commands directly")
        sub_process = False

    if copy_files and playground_path:
        if os.path.exists(os.path.join(playground_path, folder_name)):
            print(f"Folder {folder_name} already exists in {playground_path}")
            return ""
        os.makedirs(os.path.join(playground_path, folder_name), exist_ok=True)

    # Run the command or just return the string
    if sub_process:
        try:
            # Make this a proper playground implementation 
            # instead of the hallucinated code from the previous edit
            cmd = f"cd {conda_base} && {cmd_or_str}"
            result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('utf-8', errors='ignore')
            return result
        except subprocess.CalledProcessError as e:
            return e.output.decode('utf-8', errors='ignore')
    else:
        return cmd_or_str

def parse_test_case(test_case: str) -> dict:
    """
    Parses a test case
    """
    test_case = test_case.strip()
    if not test_case:
        return {"success": False}
    # Try to run the test case
    try:
        # Try running the test case
        result = make_conda_playground(test_case, log_only=False, copy_files=False)

        # Check if there are import or other errors
        test_case_lines = [line for line in test_case.split("\n") if line.strip()]
        err_lines = [line for line in result.split("\n") if "rror" in line]
        for i in range(len(test_case_lines)):
            result_index = i
            if result_index >= len(test_case_lines):
                break
            if test_case_lines[i] in err_lines:
                print(f"ERROR: {test_case_lines[i]}")
                if "ImportError" in test_case_lines[i]:
                    # Try to fix import error
                    module = re.search(r"ImportError: No module named '(.*)'", test_case_lines[i])
                    if module:
                        module = module.group(1)
                        print(f"Trying to install {module}")
                        os.system(f"cd {Config.conda_base} && pip install {module}")
        return {"success": True, "test_case": test_case}
    except Exception as e:
        print(f"Error: {e}")
        return {"success": False}

def get_test_case_descriptions(test_instances: Dict[str, Any], output_file: str) -> List[Dict[str, Any]]:
    """
    Get test case descriptions for a list of test instances
    """
    # Get configuration from Config class
    desc_model = Config.Description.model
    base_url = Config.Description.base_url
    
    # Set up logger
    logger = logging.getLogger(__name__)
    
    total_test_instances = len(test_instances)
    print(f"Getting test case descriptions for {total_test_instances} test instances")
    
    results = []
    for function_id, instance in tqdm(test_instances.items(), total=total_test_instances):
        # Use the configured settings for API calls
        result = get_openai_response(
            messages=[{"role": "user", "content": SUMMARIZE_GHERKIN_TEST.format(instance["repo"], instance["problem_statement"], instance["patch"], instance["hints_text"])}],
            max_tokens=2048,
            model=desc_model,
            base_url=base_url,
            logger=logger
        )
        if result == "Error":
            print("Too long when generating raw desc")
            result = get_openai_response(
                messages=[{"role": "user", "content": SUMMARIZE_GHERKIN_TEST.format(instance["repo"], instance["problem_statement"], instance["patch"], "No Hints Text Provided")}],
                max_tokens=2048,
                model=desc_model,
                base_url=base_url,
                logger=logger
            )     
        
        desc, _ = get_openai_response(
            messages=[{"role": "user", "content": MAKE_GHERKIN_TEST.format(instance["repo"], instance["problem_statement"], instance["patch"], instance["hints_text"], result)}],
            max_tokens=2048,
            model=desc_model,
            base_url=base_url,
            logger=logger
        )
        if desc == "Error":
            print("Too long when generating desc")
            desc, _ = get_openai_response(
                messages=[{"role": "user", "content": SUMMARIZE_GHERKIN_TEST.format(instance["repo"], instance["problem_statement"], instance["patch"], "No Hints Text Provided")}],
                max_tokens=2048,
                model=desc_model,
                base_url=base_url,
                logger=logger
            )
        pattern = r'```(?:gherkin\n|\n)(.*?)\n```'
        descs = re.findall(pattern, desc, re.DOTALL)

        results.append({
            "repo": instance["repo"],
            "instance_id": instance["instance_id"],
            "problem_statement": instance["problem_statement"],
            "patch": instance["patch"],
            "created_at": instance["created_at"],
            "hints_text": instance["hints_text"],
            "base_commit": instance["base_commit"],
            "descs": descs,
            "model": desc_model
        })

    return results
