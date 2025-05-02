import argparse
import ast
import concurrent
import json
import logging
import os
import random
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import jsonlines
from src.localizer.get_repo_structure import clone_repo
from src.utils.utils import (extract_test_patch, get_environment_yml,
                             get_requirements)
from tqdm import tqdm
from src.config import Config

DEBUG = False

def is_test_folder_empty(folder_path):
    try:
        result = subprocess.run(
            ["pytest", "--collect-only", folder_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if "collected 0 items" in result.stdout:
            return True
        else:
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error running pytest: {e}")
        return False

def setup_env(env_name, repo_path, testcases, logger=None):
    if testcases:
        try:
            env_code = (
                "\n".join([testcase["env"] for testcase in testcases])
                .strip()
                .replace("\n", " ; \\\n")
                .replace('"', "'")
            )
            env_code = env_code.replace("pip", f'{Config.conda_base}/envs/{env_name}/bin/python -m pip')
            env_code = f'{Config.conda_bin} run -n {env_name} bash -c "{env_code}"'
            subprocess.run(
                env_code, 
                shell=True, 
                text=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                cwd=repo_path
            )
            # logger.info(f'ENV_CODE: {env_code}; {result}')
            
        except Exception as e:
            if logger:
                logger.error(f"Exception during environment setup: {str(e)}")
    try:
        subprocess.run(
            f'{Config.conda_base}/envs/{env_name}/bin/python -m pip install --force-reinstall -e .; {Config.conda_base}/envs/{env_name}/bin/python -m pip install -e .[test];{Config.conda_base}/envs/{env_name}/bin/python -m pip install -e .[testing];{Config.conda_base}/envs/{env_name}/bin/python -m pip install -e .[tests];{Config.conda_base}/envs/{env_name}/bin/python -m pip install -e .[dev]',
            cwd=repo_path,
            capture_output=True,
            text=True,
            shell=True
        )
        # logger.critical(f'Dependency result: {result}')
    except Exception as e:
        if logger:
            logger.error(f"Error add dependencies: {str(e)}")

def setup_env_swe(instance, env_name, repo_path, logger=None, install_deps=False):
    if install_deps:
        result = get_requirements(instance, repo_path, logger)
        if not result:
            get_environment_yml(instance, env_name, repo_path)
        if os.path.exists(os.path.join(repo_path, "environment.yml")):
            subprocess.run(
                f'rm -rf {Config.conda_base}/envs/{env_name}',
                cwd=repo_path,
                capture_output=True,
                text=True,
                shell=True
            )
            result = subprocess.run(
                f'conda env create -f environment.yaml -n {env_name}',
                cwd=repo_path,
                capture_output=True,
                text=True,
                shell=True
            )
        elif os.path.exists(os.path.join(repo_path, "requirements.txt")): 
            try:
                result = subprocess.run(
                    f'{Config.conda_base}/envs/{env_name}/bin/python -m pip install -r requirements.txt',
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    shell=True
                )
            except Exception as e:
                pass
    try:
        result = subprocess.run(
            f'{Config.conda_base}/envs/{env_name}/bin/python -m pip install --force-reinstall -e .; {Config.conda_base}/envs/{env_name}/bin/python -m pip install -e .[test];{Config.conda_base}/envs/{env_name}/bin/python -m pip install -e .[testing];{Config.conda_base}/envs/{env_name}/bin/python -m pip install -e .[tests];{Config.conda_base}/envs/{env_name}/bin/python -m pip install -e .[dev]',
            cwd=repo_path,
            capture_output=True,
            text=True,
            shell=True
        )
        # logger.critical(f'Dependency result: {result}')
    except Exception as e:
        if logger:
            logger.error(f"Error add dependencies: {str(e)}")
    return result

def setup_logging(output_folder):    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(__name__)

def check_conda_env_exists(env_name: str):
    try:
        result = subprocess.run(f"{Config.conda_bin} env list", text=True, capture_output=True, shell=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if env_name in line:
                    return True
        else:
            return False
    except Exception as e:
        return False

def reset(repo, repo_playground, commit_id, logger, remove=False):
    repo_name = repo.split("/")[-1]
    repo_path = f'{repo_playground}/{repo_name}'

    def reset_repo():
        try:
            # ?? .pyc ??? __pycache__ ??
            clean_commands = [
                ['find', repo_path, '-name', '*.pyc', '-exec', 'rm', '-f', '{}', ';'],
                ['find', repo_path, '-name', '__pycache__', '-prune', '-exec', 'rm', '-r', '{}', ';']
            ]

            # ??????
            for cmd in clean_commands:
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    logger.error(f"Failed to execute clean command: {' '.join(cmd)}: {result.stderr.decode().strip()}")

            # Git ??
            commands = [
                ['git', 'add', '.'],
                ['git', 'stash'],
                ['git', 'stash', 'clear'],
                ['git', 'checkout', commit_id]
            ]
            returnCode = 1
            for cmd in commands:
                result = subprocess.run(cmd, cwd=repo_path, capture_output=True)
                if result.returncode != 0:
                    logger.error(f"Failed to execute at {repo_path}, {' '.join(cmd)}: {result.stderr.decode().strip()}")
                    return result.returncode
            return returnCode or result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"An error occurred: {e}")
            return False
        
    if (not os.path.exists(repo_path) or len(os.listdir(repo_path)) <= 1) or (remove and os.path.exists(repo_path)):
        if os.path.exists(repo_playground):
            os.system(f'rm -rf {repo_playground}')
        os.makedirs(repo_playground, exist_ok=True)
        clone_repo(repo, repo_playground, logger)
    
    reset_repo()

def clean_runtime(repo_playground):
    os.system(f"rm -rf {repo_playground}")

def run_tests(repo, repo_playground, env_name, testcases, correct_only=False, given_testcase=False, logger=None, build_phase=False, incorrect_only=False, midfix=None):
    repo_name = repo.split("/")[-1]
    repo_path = os.path.join(repo_playground, repo_name)
    os.makedirs(repo_playground, exist_ok=True)
    os.makedirs(repo_path, exist_ok=True)
    results = []
    corrected = False
    if midfix == None:
        midfixs = [midfix]
    else:
        midfix = ""
        midfixs = ['']
    if not given_testcase:
        for idx, testcase in enumerate(testcases):
            test_name = f'swedev_test.py'
            success = 0
            for midfix in midfixs:
                if midfix:
                    if not os.path.exists(os.path.join(repo_path, midfix)):
                        os.makedirs(os.path.join(repo_path, midfix), exist_ok=True)
                    test_path = os.path.join(repo_path, midfix, test_name)
                else:
                    test_path = os.path.join(repo_path, test_name)
                write_file(test_path, testcase['content'])
                result = {}
                try:
                    run_code = f'{Config.conda_base}/envs/{env_name}/bin/python -m pytest --no-header -rA -p no:cacheprovider -W ignore::DeprecationWarning --continue-on-collection-errors --tb=short --json={repo_path}/report.json {test_path}'
                    process = subprocess.run(
                        run_code,
                        cwd=repo_path,
                        capture_output=True, 
                        text=True,
                        shell=True
                    )
                    output = process.stdout
                    error_output = process.stderr
                    exit_code = process.returncode
                    logger.debug(output)
                    
                    test_functions = {}
                    with open(f"{repo_path}/report.json", "r") as f:
                        report = json.load(f)
                    for test in report['report']['tests']:
                        func_name, test_outcome = test['name'], test['outcome']
                        test_functions[func_name] = test_outcome

                    if correct_only:
                        if any(exec_result == 'FAILED' for exec_result in test_functions.values()):
                            with open(test_path, 'r') as f:
                                content = f.read()
                            tree = ast.parse(content)
                            failed_functions = []
                            for func_name, func_result in test_functions.items():
                                if func_result == 'FAILED':
                                    failed_functions.append(func_name)
                            for func_name in failed_functions:
                                del test_functions[func_name]
                            new_body = []
                            for node in tree.body:
                                if isinstance(node, ast.FunctionDef) and node.name in failed_functions:
                                    continue
                                new_body.append(node)
                            new_tree = ast.Module(body=new_body, type_ignores=[])
                            new_code = ast.unparse(new_tree)

                            with open(test_path, 'w') as f:
                                f.write(new_code)
                            result["content"] = new_code

                            output = "PASSED: manually exclude wrong ones" + output
                            error_output = "PASSED: manually exclude wrong ones" + error_output
                            exit_code = 0
        
                    if incorrect_only:
                        if any(exec_result == 'PASSED' for exec_result in test_functions.values()):
                            corrected = True
                            with open(test_path, 'r') as f:
                                content = f.read()
                            tree = ast.parse(content)
                            passed_functions = {func_name for func_name, result in test_functions.items() 
                                                if result == 'PASSED'}
                            new_body = []
                            for node in tree.body:
                                if isinstance(node, ast.FunctionDef) and node.name in passed_functions:
                                    continue
                                new_body.append(node)
                            new_tree = ast.Module(body=new_body, type_ignores=[])
                            new_code = ast.unparse(new_tree)

                            with open(test_path, 'w') as f:
                                f.write(new_code)
                            result["content"] = new_code
                            output = "FAILED: manually exclude correct ones" + output
                            error_output = "FAILED: manually exclude correct ones" + error_output
                            exit_code = 0

                    if "no tests ran" in output.lower() or "no tests ran" in error_output.lower():
                        result["status"] = "empty"
                    elif "PASSED: manually exclude wrong ones" in output:
                        result["status"] = "success"
                    elif exit_code == 0 and not "ERROR" in output and not "FAILED" in output \
                            and not "ERROR" in error_output and not "FAILED" in error_output \
                            and ("pass" in error_output.lower() or "pass" in output.lower()):
                        result["status"] = "success"
                    elif "FAILED: manually exclude correct ones" in output:
                        result["status"] = "failed"
                    elif exit_code == 0 and "fail" in output.lower():
                        result["status"] = "failed"
                    else:
                        result["status"] = "error"
                        
                    logger.critical(f'{repo}: {result["status"]}')
                    result["output"] = {
                        "stdout": output,
                        "stderr": error_output,
                        "exit_code": exit_code
                    }

                    result["functions"] = test_functions
                    if os.path.exists(test_path):
                        os.remove(test_path)

                except Exception as e:
                    if os.path.exists(test_path):
                        os.remove(test_path)
                    logger.error(f"Error running test case {idx}: {str(e)}")
                    result["status"] = "error"
                    result["output"] = {
                        "error": str(e),
                        "type": type(e).__name__
                    }
                finally:
                    os.remove(test_path)
                if result['status'] == 'error':
                    continue
                if not "content" in result:
                    result["content"] = testcase["content"]
                results.append(result)
                success = 1
                break
                logger.info(f"Test case {idx}({'w/ patch' if correct_only or build_phase else 'wo/ patch'}) completed with status: {result['status']}")
                
            if not success:
                if not "content" in result:
                    result["content"] = testcase["content"]
                results.append(result)
    return results, midfix

def get_test_files_content(test_folder='tests', logger=None):
    test_files_content = []
    file_id = 0
    for root, _, files in os.walk(test_folder):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    test_files_content.append({
                        'content': content
                    })
                    file_id += 1
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
    return test_files_content

def init_env(env_name, repo, repo_playground, commit_id, testcases=None, logger=None):
    reset(repo, repo_playground, commit_id, logger, remove=False)    
    if check_conda_env_exists(env_name):
        pass
    else:
        try:
            subprocess.run(
                f'{Config.conda_bin} create -n {env_name} --clone swedevbase -y',
                capture_output=True,
                text=True,
                shell=True
            )
        except Exception as e:
            logger.error(f"Error creating conda environment: {str(e)}")
            pass

def write_file(file_path, content, cwd=None, extra_conmmands=None, logger=None):
    with open(file_path, "w") as f:
        f.write(content)
    if extra_conmmands:
        try:
            if 'patch.diff' in file_path and not 'test_patch.diff' in file_path:
                subprocess.run(
                    f"sed -i 's/\\r//g' {file_path}",
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    shell=True
                )

            result = subprocess.run(
                extra_conmmands,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
                shell=True
            )
        except Exception as e:
            logger and logger.critical(f'File at {cwd}, {file_path} apply failed, {e}')
            pass
    

def evaluate_testcase(instance, given_testcase=None, eval_mode=False, logger=None) -> dict:
    SWE_FORMAT = "test_patch" in instance.keys() and instance["test_patch"]
    repo = instance["repo"]
    instance_id = instance["instance_id"]
    commit_id = instance["base_commit"]
    patch = instance["patch"] if not eval_mode else instance["solution"]
    testcases = instance["tests"] if "tests" in instance.keys() else None
    test_patch = instance["test_patch"] if "test_patch" in instance.keys() else None

    logger.info(f"\nEvaluating testcase for {repo}; Instance ID: {instance_id}; Commit ID: {commit_id}. Testcases: {len(testcases) if testcases else None}, {instance.keys()}")
    if not testcases and not test_patch:
        # testcases = [{"env": "", "content": ""}]
        logger.error("No testcases provided. Exiting...")
        return None

    repo_id = f'{instance_id}_{repo.replace("/", "_")}_{commit_id}'
    repo_name = repo.split("/")[-1]
    repo_playground = os.path.join(Config.playground_path, repo_id)
    repo_path = os.path.join(repo_playground, repo_name)
    env_name = f'swedev_{instance_id}'
    clone_repo(repo, repo_playground, logger)
    init_env(env_name, repo, repo_playground, commit_id, testcases, logger)

    if is_test_folder_empty(repo_playground) and not test_patch and not testcases:
        return None
    
    os.makedirs(repo_playground, exist_ok=True)
    os.makedirs(repo_path, exist_ok=True)
    
    if SWE_FORMAT and not testcases:
        logger.critical("Enable SWE_FORMAT")
        testcases = []
        test_contents = extract_test_patch(repo_path, test_patch)
        for _, content in enumerate(test_contents.values()):
            testcases.append({
                "env": "",
                "content": content
            })
            
    if not testcases:
        return None

    if SWE_FORMAT:
        setup_env_swe(
            instance,
            env_name,
            repo_path,
            logger,
            install_deps=True
        )
    else:
        setup_env(env_name, repo_path, testcases, logger)

    results_without_patch, midfix = run_tests(repo, repo_playground, env_name, testcases, correct_only=False, incorrect_only=False, given_testcase=given_testcase, logger=logger)
    assert len(results_without_patch) == len(testcases)

    reset(repo, repo_playground, commit_id, logger, remove=False)
    if SWE_FORMAT:
        setup_env_swe(
            instance,
            env_name,
            repo_path,
            logger,
            install_deps=True
        )
    else:
        setup_env(env_name, repo_path, testcases, logger)
    
    # apply patch
    patch_path = os.path.join(repo_path, "patch.diff")
    exec_command = f"git apply -v {patch_path} || patch --batch --fuzz=5 -p1 -i {patch_path}"
    write_file(patch_path, patch.replace("\\n", "\n"), cwd=repo_path, extra_conmmands=exec_command, logger=logger)
    correct_only = False
    incorrect_only = False
    results_with_patch, _ = run_tests(repo, repo_playground, env_name, testcases, correct_only=correct_only, incorrect_only=incorrect_only, given_testcase=given_testcase, logger=logger, midfix=midfix)
    assert len(results_with_patch) == len(testcases)

    results = []
    for i in range(len(testcases)):
        result = {}
        result["content"] = testcases[i]["content"]
        result["env"] = testcases[i]["env"]
        FAIL_TO_PASS = []
        if 'functions' in results_with_patch[i].keys():
            try:
                for func_wpatch, result_wpatch in results_with_patch[i]['functions'].items():
                    if not result_wpatch == 'passed':
                        continue
                    if not func_wpatch in results_without_patch[i]['functions'].keys():
                        logger.info(f'Func {func_wpatch} does not appear in results without patch.')
                        FAIL_TO_PASS.append({"name": func_wpatch, "status": "not appear in results without patch"})
                    elif not results_without_patch[i]['functions'][func_wpatch] == 'passed':
                        FAIL_TO_PASS.append({"name": func_wpatch, "status": "normal"})
                result["FAIL_TO_PASS"] = FAIL_TO_PASS
                result["wpatch_funcs"] = results_with_patch[i]['functions']
                result["wopatch_funcs"] = results_without_patch[i]['functions']
                result['wpatch_log'] = results_with_patch[i]
                result['wopatch_log'] = results_without_patch[i]
            except:
                results.append(results_without_patch[i])
        results.append(result)
    return results
    
def process_single_item(item: dict, given_testcase: bool, eval_mode: bool, output_file, logger) -> dict:
    try:
        instance_id = item["instance_id"]
        base_commit = item["base_commit"]
        if base_commit is None:
            return None

        result = evaluate_testcase(
            item,
            given_testcase,
            eval_mode=eval_mode,
            logger=logger
        )

        if result:
            logger.info(f"Evaluation completed successfully for instance {instance_id}")
        else:
            logger.warning(f"Evaluation returned no results for instance {instance_id}")

        item['test'] = result
        try:
            with open(output_file, "a") as f:
                f.write(json.dumps(item) + "\n")
        except Exception as e:
            pass
        if os.path.exists(f'{Config.conda_base}/envs/swedev_{instance_id}'):
            os.system(f'rm -rf {Config.conda_base}/envs/swedev_{instance_id}')
        repo_id = f'{instance_id}_{item["repo"].replace("/", "_")}_{item["base_commit"]}'
        repo_playground = os.path.join(Config.playground_path, repo_id)
        if os.path.exists(repo_playground):
            os.system(f'rm -rf {repo_playground}')
        return item

    except Exception as e:
        traceback.print_exc()
        if os.path.exists(f'{Config.conda_base}/envs/swedev_{instance_id}'):
            os.system(f'rm -rf {Config.conda_base}/envs/swedev_{instance_id}')
        repo_id = f'{instance_id}_{item["repo"].replace("/", "_")}_{item["base_commit"]}'
        repo_playground = os.path.join(Config.playground_path, repo_id)
        if os.path.exists(repo_playground):
            os.system(f'rm -rf {repo_playground}')
        return None

def process_single_item_with_timeout(item: dict, given_testcase: bool, eval_mode, output_file, logger, timeout=900) -> dict:
    # Timeout is set to 15 minutes (900 seconds) by default
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process_single_item, item, given_testcase, eval_mode, output_file, logger)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None
        
def save_result(result: dict, output_file: str, logger):
    try:
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--show_report", action="store_true")
    parser.add_argument("--given_testcase", action="store_true")
    parser.add_argument("--eval_mode", action="store_true")
    parser.add_argument("--output_file", type=str, default="evaluated_testcases.jsonl")

    args = parser.parse_args()
    
    logger = setup_logging(args.output_folder)
    logger.info("Starting test execution process")
    
    if args.show_report:
        logger.info("Generating report from results")
        useful_ds = []
        results = []
        with open(args.dataset, 'r') as f:
            for line in f.readlines():
                results.append(json.loads(json.dumps(eval(line))))

        total, fail2pass, error2pass, pass2fail, pass2error, pass2pass = 0, 0, 0, 0, 0, 0
        fail2fail, fail2error, error2fail, error2error = 0, 0, 0, 0
        used_repos = set()
        used_instances = set()
        all_repos = set()
        all_instances = set()
        for result in results:
            tests = result["tests"]
            results_without_patch = tests["results_without_patch"]
            results_with_patch = tests["results_with_patch"]
            
            for i in range(len(results_without_patch)):
                all_repos.add(result["repo"])
                all_instances.add(result["instance_id"])
                result_without_patch = results_without_patch[i]
                result_with_patch = results_with_patch[i]
                if result_with_patch["content"] == 'i':
                    continue

                # Fail/Error -> Success
                if (result_without_patch["status"] == "failed" or result_without_patch["status"] == "error") and result_with_patch["status"] == "success":
                    if result_without_patch["status"] == "failed":
                        fail2pass += 1
                    elif result_without_patch["status"] == "error":
                        error2pass += 1
                    used_repos.add(result["repo"])
                    used_instances.add(result["instance_id"])
                    useful_ds.append(result)

                # Success -> Fail/Error
                elif result_without_patch["status"] == "success" and (result_with_patch["status"] == "failed" or result_with_patch["status"] == "error"):
                    if result_with_patch["status"] == "failed":
                        pass2fail += 1
                    elif result_with_patch["status"] == "error":
                        pass2error += 1
                    logger.debug("Found pass to fail/error case:")
                    logger.debug(f"Before patch: {result_without_patch}")
                    logger.debug(f"After patch: {result_with_patch}")

                # Success -> Success
                elif result_without_patch["status"] == "success" and result_with_patch["status"] == "success":
                    pass2pass += 1

                # Fail -> Fail
                elif result_without_patch["status"] == "failed" and result_with_patch["status"] == "failed":
                    fail2fail += 1

                # Error -> Error
                elif result_without_patch["status"] == "error" and result_with_patch["status"] == "error":
                    error2error += 1

                # Fail -> Error
                elif result_without_patch["status"] == "failed" and result_with_patch["status"] == "error":
                    fail2error += 1

                # Error -> Fail
                elif result_without_patch["status"] == "error" and result_with_patch["status"] == "failed":
                    error2fail += 1
                total += 1

        logger.info(f"Total runnable: {total}")
        logger.info(f"Fail2Pass: {fail2pass}")
        logger.info(f"Error2Pass: {error2pass}")
        logger.info(f"Pass2Fail: {pass2fail}")
        logger.info(f"Pass2Error: {pass2error}")
        logger.info(f"Pass2Pass: {pass2pass}")
        logger.info(f"Fail2Fail: {fail2fail}")
        logger.info(f"Fail2Error: {fail2error}")
        logger.info(f"Error2Fail: {error2fail}")
        logger.info(f"Error2Error: {error2error}")
        logger.info(f"Related repos: {len(used_repos)}")
        logger.info(f"Related instances: {len(used_instances)}")
        logger.info(f"Total repos: {len(all_repos)}")
        logger.info(f"Total instances: {len(all_instances)}")
        
        with open(f"{'/'.join(args.dataset.split('/')[:-1])}/fail2pass.jsonl", 'w') as f:
            for item in used_instances:
                f.write(json.dumps(item) + '\n')
        exit()

    os.makedirs(args.output_folder, exist_ok=True)
    args.output_file = os.path.join(args.output_folder, args.output_file)
    
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = []
    if args.dataset.endswith('.json'):
        with open(args.dataset, 'r') as f:
            dataset = json.load(f)
    elif args.dataset.endswith('.jsonl'):
        with open(args.dataset, 'r', encoding='latin-1') as f:
            for line in f:
                try:
                    dataset.append(json.loads(line))
                except Exception as e:
                    print('error when parseing', e)
    else:
        logger.error("Dataset file must be in JSON or JSONL format")
        raise ValueError("Dataset file must be in JSON or JSONL format")

    print(len(dataset))
    if os.path.exists(args.output_file):
        with jsonlines.open(args.output_file, 'r') as f:
            used = [line for line in f]
            commits = [d['base_commit'] for d in used]
        commits = list(set(commits))
        print(len(commits))
        dataset = [d for d in dataset if not d['base_commit'] in commits]

    logger.info(f"Processing {len(dataset)} items with {args.num_workers} workers")
    dataset = list(reversed(dataset))
    random.shuffle(dataset)
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_single_item_with_timeout, item, args.given_testcase, args.eval_mode, args.output_file, logger)
            for item in dataset
        ]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing testcases"):
            pass

    logger.info("All tasks completed")
    
if __name__ == "__main__":
    main()