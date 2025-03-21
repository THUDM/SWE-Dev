import base64
import hashlib
import json
import os
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path

import requests
from modelscope import AutoTokenizer
import os

SWE_BENCH_URL_RAW = "https://raw.githubusercontent.com/"

def get_environment_yml(
        instance: dict,
        env_name: str,
        save_path: str = None,
        python_version: str = None,
    ) -> str:
    """
    Get environment.yml for given task instance.

    Args:
        instance (dict): SWE Bench Task instance (unused in this version; kept for compatibility).
        env_name (str): Rename retrieved environment.yml to this name.
        save_path (str): If provided, save environment.yml to this path.
        python_version (str): Python version to include in the environment.yml.
    Returns:
        environment.yml (str): If save_path given, returns path to saved environment.yml.
            Otherwise, returns environment.yml as string.
    """
    if save_path is None or not os.path.isdir(save_path):
        raise ValueError("save_path must be a valid directory.")

    # Find all YAML files containing 'environment' in their name
    env_files = [
        os.path.join(save_path, f)
        for f in os.listdir(save_path)
        if "environment" in f.lower() and f.endswith(".yml")
    ]

    if not env_files:
        # No environment.yml files found
        return None

    combined_lines = []
    for env_file in env_files:
        try:
            with open(env_file, "r") as f:
                combined_lines.extend(f.readlines())
        except Exception as e:
            # Log or print errors if necessary
            continue

    # Process and clean the environment.yml content
    cleaned = []
    dependencies_added = False
    unique_lines = set()  # To remove duplicates

    for line in combined_lines:
        line = line.strip()  # Remove leading/trailing whitespace
        # Skip empty lines and duplicates
        if not line or line in unique_lines:
            continue
        unique_lines.add(line)

        # Rename the environment if "name:" is found
        if line.startswith("name:"):
            cleaned.append(f"name: {env_name}")
            continue

        # Add python version if "dependencies:" is found
        if line.startswith("dependencies:"):
            cleaned.append(line)
            if python_version is not None and not dependencies_added:
                cleaned.append(f"  - python={python_version}")
                dependencies_added = True
            continue

        # Append all other lines
        cleaned.append(line)

    # Return the cleaned environment.yml string if no save path is given
    if save_path is None:
        return "\n".join(cleaned)

    # Save the cleaned environment.yml to the specified path
    path_to_env = os.path.join(save_path, "environment.yml")
    try:
        with open(path_to_env, "w") as f:
            f.write("\n".join(cleaned))
    except Exception as e:
        raise RuntimeError(f"Error saving environment.yml: {str(e)}")

    return path_to_env

def get_requirements(instance: dict, save_path: str = None, logger=None):
    """
    Get requirements.txt for given task instance.

    Args:
        instance (dict): task instance
        save_path (str): Directory to search for requirements files and optionally save the final requirements.txt.
    Returns:
        requirements.txt (str): If save_path given, returns path to saved requirements.txt.
            Otherwise, returns requirements.txt as a string.
    """
    if save_path is None or not os.path.isdir(save_path):
        raise ValueError("save_path must be a valid directory.")

    requirements_files = [
        os.path.join(save_path, f)
        for f in os.listdir(save_path)
        if f.lower().startswith("requirements") and f.endswith(".txt")
    ]

    if not requirements_files:
        if logger:
            logger.warning("No requirements files found in the provided save_path.")
        # Return None if no requirements files are found
        return None

    combined_requirements = []
    exclude_line = lambda line: any(
        [line.strip().startswith(x) for x in ["-e .", "#", ".[test"]]
    )

    for req_file in requirements_files:
        try:
            with open(req_file, "r") as f:
                for line in f:
                    line = line.strip()  # Remove leading/trailing whitespace
                    if line.startswith("-r"):
                        # Handle recursive requirements (look for referenced files in the same directory)
                        referenced_file = line[len("-r"):].strip()
                        ref_path = os.path.join(save_path, referenced_file)
                        if os.path.isfile(ref_path):
                            with open(ref_path, "r") as ref_f:
                                combined_requirements.extend(
                                    l.strip() for l in ref_f if not exclude_line(l.strip())
                                )
                        else:
                            if logger:
                                logger.warning(f"Referenced file {ref_path} not found.")
                    elif not exclude_line(line):
                        combined_requirements.append(line)
        except Exception as e:
            if logger:
                logger.error(f"Error reading {req_file}: {str(e)}")
            continue

    combined_requirements = [line for line in combined_requirements if line and not 'git' in combined_requirements]
    all_reqs = "\n".join(sorted(set(filter(None, combined_requirements))))
    if save_path is None:
        return all_reqs

    # Save the combined requirements to a new requirements.txt file in save_path
    path_to_reqs = os.path.join(save_path, "requirements.txt")
    try:
        with open(path_to_reqs, "w") as f:
            f.write(all_reqs)
        if logger:
            logger.info(f"Combined requirements.txt saved at {path_to_reqs}")
    except Exception as e:
        if logger:
            logger.error(f"Error saving combined requirements.txt: {str(e)}")
        raise e

    return path_to_reqs

def generate_hash(text):
    hash_object = hashlib.sha256()
    text_bytes = text.encode('utf-8')
    hash_object.update(text_bytes)
    hash_bytes = hash_object.digest()
    hash_base64 = base64.b64encode(hash_bytes)
    hash_str = hash_base64.decode('utf-8').replace('+', '0').replace('/', '0').replace('=', '')
    return hash_str

def calc_cost(input_tokens, output_tokens): 
    return input_tokens + output_tokens

def call(
    model: str = None,
    base_url: str = None,
    messages: list[dict] = [],
    temperature: float = 0.5,
    max_tokens: int = 200,
    top_p: float = 0.95,
    tools: list[dict] | None = None,
    stop: list[str] = ['<|user|>'],
    platform: str = 'openai',
    proxies: dict | None = None,
    logger=None,
    **kwargs
):
    if len(messages[0]['content']) > 200000:
        return "Error", {}
    api_key = os.environ.get("OPENAI_API_KEY", "sk-test")
    if not model:
        model = os.environ.get("OPENAI_BASE_MODEL", 'gpt-4o')
        base_url = os.environ.get("OPENAI_BASE_URL", 'http://api.openai.com/v1')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    if platform == 'openai':
        url = f'{base_url}/chat/completions'
        data = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'stream': False,
            'stop': stop,
        }
        for retry in range(3):
            if retry > 1:
                print(f'retry: {retry}')
            try:
                response = requests.post(url, json=data, headers=headers, proxies=proxies)
                response = response.json()
                if not 'choices' in response.keys():
                    logger and logger.critical(response)
                content = response["choices"][0]["message"]["content"]

                metrics = {} # dummy metrics
                return content, metrics
            except Exception as e:
                logger and logger.critical(traceback.print_exc())
                time.sleep(5)
        return "Error", {}
    else:
        raise NotImplementedError
    
def combine_by_instance_id(data):
    """
    Combine data entries by their instance ID.

    Arguments:
    data -- a list of dictionaries with instance IDs and other information

    Returns:
    A list of combined dictionaries by instance ID with all associated data.
    """
    combined_data = defaultdict(lambda: defaultdict(list))
    for item in data:
        instance_id = item.get("instance_id")
        if not instance_id:
            continue
        for key, value in item.items():
            if key != "instance_id":
                combined_data[instance_id][key].extend(
                    value if isinstance(value, list) else [value]
                )
    return [
        {**{"instance_id": iid}, **details} for iid, details in combined_data.items()
    ]

def extract_test_patch(repo_path, test_patch):
    test_files = {}
    modified_files = set()
    for line in test_patch.split('\n'):
        if line.startswith('--- a/') or line.startswith('+++ b/'):
            file_path = line[6:]
            if "test" in file_path and file_path.endswith('.py'):
                modified_files.add(file_path)
    patch_path = os.path.join(repo_path, 'testcase.patch')
    with open(patch_path, 'w') as f:
        f.write(test_patch)
        
    for file_path in modified_files:
        full_path = os.path.join(repo_path, file_path)
        dir_path = os.path.dirname(full_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            Path(full_path).touch()
            
    try:
        subprocess.run(
            ['git', 'apply', '--whitespace=nowarn', patch_path],
            capture_output=True,
            text=True,
            cwd=repo_path
        )

        for file_path in modified_files:
            full_path = os.path.join(repo_path, file_path)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    content = f.read()
                test_files[file_path] = content
    except subprocess.CalledProcessError as e:
        print(f"Error applying patch: {e}")
    return test_files