import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List

import jsonlines
from src.localizer.get_repo_structure import clone_repo
from src.localizer.localize import get_location
from src.testcases.eval_testcases import init_env, run_tests, setup_env
from src.utils.error_handler import *
from src.utils.extract_signs import *
from src.utils.prompts import *
from src.utils.utils import *
from tqdm import tqdm
from src.config import CONDA_BASE, PLAYGROUND_PATH, OPENAI_BASE_MODEL, OPENAI_BASE_URL, get_config_value

DEBUG = False
REVISE_ROUNDS = 0

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

def get_installed_apt_packages():
    try:
        result = subprocess.run(['dpkg', '-l'], capture_output=True, text=True)
        packages = []
        for line in result.stdout.splitlines():
            if line.startswith('ii'):
                packages.append(line.split()[1].split(':')[0])
        return set(packages)
    except:
        return set()

def get_installed_pip_packages(pip_path):
    try:
        result = subprocess.run([pip_path, 'list'], capture_output=True, text=True)
        packages = []
        for line in result.stdout.splitlines()[2:]:  # Skip header rows
            packages.append(line.split()[0].lower())
        return set(packages)
    except:
        return set()

def parse_testcase(text, pip_path):
    # Cache installed packages
    installed_apt_packages = get_installed_apt_packages()
    installed_pip_packages = get_installed_pip_packages(pip_path)
    
    commands = [
        'rm', 'git clone', 'dd', 'mkfs', 'shutdown', 'reboot', 'kill', 'mv', 'scp', 'ifconfig',
        'rsync', 'pkill', 'docker rm', 'docker rmi', 'iptables', 'ufw', 'mount', 'umount',
    ]    
    install_commands = ['pip install', 'apt-get', 'apt', 'echo', 'touch', 'mkdir', 'mv', 'cp', 'cat']
    
    python_stdlib = [
        'abc', 'argparse', 'array', 'ast', 'asyncio', 'base64', 'bisect', 'calendar',
        'collections', 'concurrent', 'configparser', 'contextlib', 'copy', 'csv', 'datetime',
        'decimal', 'difflib', 'enum', 'fileinput', 'fnmatch', 'fractions', 'functools',
        'glob', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'importlib', 'inspect', 'io',
        'itertools', 'json', 'logging', 'math', 'multiprocessing', 'operator', 'os', 'pathlib',
        'pickle', 'pprint', 'random', 're', 'shutil', 'signal', 'socket', 'sqlite3', 'statistics',
        'string', 'subprocess', 'sys', 'tempfile', 'threading', 'time', 'timeit', 'tkinter',
        'traceback', 'typing', 'unittest', 'urllib', 'uuid', 'warnings', 'weakref', 'xml',
        'zipfile'
    ]

    def extract_content(tag, text):
        try:
            pattern = fr"<{tag}>(.*?)</{tag}>"
            matches = re.findall(pattern, text, re.DOTALL)
            return matches
        except:
            return []

    def extract_code_blocks(content):
        try:
            pattern = r"```(?:[\w]*)\n(.*?)```"
            return re.findall(pattern, content, re.DOTALL)    
        except:
            return []
    
    def extract_imports(code):
        try:
            pattern = r"^(?:import\s+([\w.]+)|from\s+([\w.]+)\s+import\s+\w+)"
            matches = re.findall(pattern, code, re.MULTILINE)
            packages = {match.split('.')[0] for pair in matches for match in pair if match}
            return [pkg for pkg in packages if pkg not in python_stdlib]
        except:
            return []

    def generate_pip_commands(imports):
        return [f"pip install {pkg}" for pkg in imports 
                if pkg.lower() not in installed_pip_packages]
    
    def process_apt_command(line):
        if line.startswith(('apt install', 'apt-get install')):
            packages = re.findall(r'install\s+(.*)', line)[0].split()
            filtered_pkgs = [pkg for pkg in packages 
                           if pkg not in installed_apt_packages]
            if filtered_pkgs:
                return f"apt install {' '.join(filtered_pkgs)}"
            return ""
        return line
    
    def replace_os_system(code):
        lines = code.splitlines()
        result = []
        
        for line in lines:
            if ("os.system" in line or "git clone" in line) and any([command in line for command in commands]):
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
                processed_line = process_apt_command(line)
                if processed_line:
                    ret += processed_line + '\n'
                else:
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
    file_count = args.top_n
    instance_id = loc["instance_id"]
    repo = loc["repo"]
    statement = loc["problem_statement"]
    patch = loc["patch"]
    env_name = f'swedev_{instance_id}'
    commit_id = loc["base_commit"]
    hints_text = loc['hints_text']
    repo_id = f'{instance_id}_{repo.replace("/", "_")}_{commit_id}'
    repo_playground = os.path.join(PLAYGROUND_PATH, repo_id)
    repo_name = repo.split("/")[-1]
    clone_repo(repo, repo_playground, logger)
    repo_path = f'{repo_playground}/{repo_name}'

    if "project_tree" in loc.keys() and "patch_blocks" in loc.keys():
        project_tree = loc['project_tree']
        patch_blocks = loc["patch_blocks"]
    else:
        location = get_location(loc)
        project_tree, patch_blocks = location["project_tree"], location["patch_blocks"]
        if not patch_blocks:
            return None

    try:
        function_dict = generate_signatures(extract_classes_and_functions_from_directory(repo_playground), type="function")
        class_dict = generate_signatures(extract_classes_and_functions_from_directory(repo_playground), type="class")

        def get_content(patch_blocks):
            content = ""
            for block in patch_blocks:
                content += f"<block>\n{block['file']}:\n\n```python\n{block['code']}\n```\n<block/>\n"
            content = content.strip() if content else "No file content provided"
            return content

        def get_raw_test(testcase=None, desc=None, revise=False, history="", with_patch=False):
            cur_file_count = file_count
            cur_project_tree = project_tree
            cur_hints_text = hints_text
            while cur_file_count >= 0:
                try:
                    if cur_file_count <= 2:
                        cur_project_tree = "No Project Tree Provided"
                    content = get_content(patch_blocks[:cur_file_count])
                    prompt = TESTCASE_GENERATION.format(repo, statement, cur_hints_text, patch, cur_project_tree, desc if desc else "Try to reproduce the results in the problem statement.", content).strip()
                    if revise:
                        if with_patch:
                            # Get testcase generation model and base URL from config
                            testcase_model = get_config_value("testcase.model", OPENAI_BASE_MODEL)
                            testcase_base_url = get_config_value("testcase.base_url", OPENAI_BASE_URL)
                            
                            resp, _ = call(
                                messages=[{"role": "user", "content": EXTRACT_API_PROMPT.format(history)}],
                                max_tokens=4096,
                                model=testcase_model,
                                base_url=testcase_base_url,
                                logger=logger                             
                            )
                            type, api = parse_api(resp)
                            if type == "function":
                                apis = api_formatter(find_top_similar_apis(api, function_dict, type="function", top_n=15))
                            elif type == "class":
                                apis = api_formatter(find_top_similar_apis(api, class_dict, type="class", top_n=15))
                            else:
                                apis = "No API provided"
                            prompt = REVISION_AFTER_PROMPT.format(repo, statement, "No Hints Text Provided", patch, cur_project_tree, content, apis, test_formatter(testcase), history)
                        else:
                            prompt = REVISION_BEFORE_PROMPT.format(repo, statement, patch, cur_project_tree, content, test_formatter(testcase))
                    
                    # Get testcase generation model and base URL from config
                    testcase_model = get_config_value("testcase.model", OPENAI_BASE_MODEL)
                    testcase_base_url = get_config_value("testcase.base_url", OPENAI_BASE_URL)
                    
                    resp, _ = call(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=4096,
                        model=testcase_model,
                        base_url=testcase_base_url,
                        logger=logger
                    )
                    if resp != "Error":
                        return resp
                    else:
                        cur_hints_text = "No Hints Text Provided"
                        cur_file_count -= 1
                except Exception as e: # We only take context errors
                    cur_file_count -= 1
            return None

        raw_tests, tests = [], []
        descs = loc["descs"]
        for desc in descs:
            resp = get_raw_test(testcase=None, desc=desc, revise=False)
            if resp:
                raw_tests.append(resp)
        resp = get_raw_test(testcase=None, desc=None, revise=False)
        if resp:
            raw_tests.append(resp)

        count = 0
        for i, raw_test in enumerate(raw_tests):
            try:
                generated_envs, generated_tests = parse_testcase(raw_test, f"{CONDA_BASE}/envs/swedev_{instance_id}/bin/pip")
                if not generated_tests:
                    continue
                for generated_env, generated_test in zip(generated_envs, generated_tests):
                    if generated_test:
                        tests.append({
                            "raw": raw_test,
                            "id": count,
                            "content": generated_test,
                            "env": generated_env,
                            "description": descs[i] if i != len(raw_tests) - 1 else None,
                        })
                        count += 1
            except Exception as e:
                continue

        logger.info(f"Finishing raw test generation. Raw test length: {len(raw_tests)}, Parsed test length: {len(tests)}")

        if REVISE_ROUNDS != 0:
            logger.critical("start initing")
            init_env(env_name, repo, repo_playground, commit_id, testcases=None, logger=logger)
            status = True
            # apply patch
            with open(os.path.join(repo_path, "patch.diff"), "w") as f:
                f.write(patch) 
            try:
                result = subprocess.run(
                    ['git', 'apply', 'patch.diff'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error applying patch: {e}")
                status = False
            if status:
                try:
                    for i, test in enumerate(tests):
                        history = ""
                        test_content = test["content"]
                        test_env = test["env"]
                        for round in range(REVISE_ROUNDS):
                            logger.critical(f"revise round: {round}")
                            setup_env(env_name, repo_path, [test], logger)
                            result = run_tests(repo, repo_playground, env_name, [test], correct_only=False, given_testcase=False, logger=logger, build_phase=True)[0]
                            if result["status"] == "success":
                                tests[i]["content"] = test_content
                                tests[i]["env"] = test_env
                                break
                            logger.info(f'Results w/ patch: {result["status"]}')
                            error_msg = result["output"]
                            # history += f"Round {round + 1}:\nTestcase:\n{test_content}\n\nErrors:\n{error_msg}\n"
                            history = f'Error: {error_msg}'
                            raw_test_content = get_raw_test(test, desc=None, revise=True, history=history, with_patch=True)
                            test_envs, test_contents = parse_testcase(raw_test_content, f"{CONDA_BASE}/envs/swedev_{instance_id}/bin/pip")
                            if test_contents:
                                tests[i]["content"] = test_contents[0]
                                tests[i]["env"] = test_envs[0]
                                test_content = test_contents[0]
                                test_env = test_envs[0]
                                test = tests[i]
                except Exception as e:
                    print(f"Error in running tests: {e}")
                    traceback.print_exc()

        os.system(f"rm -rf {repo_playground}")
        os.system(f"rm -rf {CONDA_BASE}/envs/swedev_{instance_id}")
        loc['tests'] = tests
        loc['revise_round'] = REVISE_ROUNDS
        return loc

    except Exception as e:
        os.system(f"rm -rf {repo_playground}")
        os.system(f"rm -rf {CONDA_BASE}/envs/swedev_{instance_id}")
        logger.error(f"Error processing instance {instance_id}: {str(e)}")
        traceback.print_exc()
        return None
            
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
        
    random.shuffle(locs)
    print(f"Total {len(locs)} instances")
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='latin-1') as f:
            # prev_o = [d for d in f]
            prev_o = []
            for d in f.readlines():
                if not d:
                    continue
                try:
                    prev_o.append(json.loads(d))
                except Exception as e:
                    # print(repr(d))
                    logging.info(e)
                    logging.info(traceback.print_exc())
        prev_o_ids = [o["instance_id"] for o in prev_o]
    else:
        prev_o_ids = []

    locs = [loc for loc in locs if loc["instance_id"] not in prev_o_ids]
    print(f"Remaining {len(locs)} instances")
    results = []
    result_lock = threading.Lock()

    def process_and_save(loc, output_file, logger):
        result = process_single_instance(loc, args, logger)
        if result:
            with result_lock:
                if result["tests"]:
                    results.append(result)
                    with open(output_file, "a") as f:
                        f.write(f'{json.dumps(result)}\n')
                else:
                    print(f"Skipping instance {loc['instance_id']} due to no tests, {result['tests']}")
                    
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
    # test_parsing(); exit()
    main()
