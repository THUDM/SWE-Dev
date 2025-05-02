import os
import subprocess
from src.utils.preprocess import parse_python_file
from src.config import LOCAL_REPO_DIR

DEBUG = False

def repo_to_top_folder(repo_name):
    """Get the top folder name from the repository name.
    """
    return repo_name.split('/')[-1]

def checkout_commit(repo_path, commit_id):
    """Checkout the specified commit in the given local git repository.
    :param repo_path: Path to the local git repository
    :param commit_id: Commit ID to checkout
    :return: None
    """
    try:
        print(f"Checking out commit {commit_id} in repository at {repo_path}...")
        subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)
        print("Commit checked out successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def clone_repo(repo, repo_playground):
    DO_CLONE = (not os.path.exists(f"{LOCAL_REPO_DIR}/{repo_to_top_folder(repo)}")) or len(os.listdir(f"{LOCAL_REPO_DIR}/{repo_to_top_folder(repo)}")) <= 1
    try:
        if DO_CLONE:
            if os.path.exists(f"{LOCAL_REPO_DIR}/{repo_to_top_folder(repo)}"):
                os.system(f'rm -rf {LOCAL_REPO_DIR}/{repo_to_top_folder(repo)}')
            for _ in range(3):
                result = subprocess.run(
                    f"git clone https://github.com/{repo}.git {LOCAL_REPO_DIR}/{repo_to_top_folder(repo)}",
                    check=True,
                    shell=True
                )
                if result.returncode == 0:
                    break
        os.makedirs(repo_playground, exist_ok=True)
        subprocess.run(
            f"cp -r {LOCAL_REPO_DIR}/{repo_to_top_folder(repo)} {repo_playground}",
            check=True,
            shell=True
        )
    except Exception as e:
        print(f"An unexpected error occurred when copying repo: {e}")

def get_project_structure_from_scratch(repo, commit_id, instance_id, repo_playground):
    """Get the project structure from scratch
    :param repo: Repository name
    :param commit_id: Commit ID
    :param instance_id: Instance ID
    :param repo_playground: Repository playground
    :return: Project structure
    """
    repo_id = f'{instance_id}_{repo.replace("/", "_")}_{commit_id}'
    repo_path = f"{repo_playground}/{repo_id}/{repo.split('/')[-1]}"
    if not os.path.exists(repo_path) or not os.path.exists(os.path.join(repo_path, "setup.py")) \
            and not os.path.exists(os.path.join(repo_path, "pyproject.toml")):
        os.makedirs(f"{repo_playground}/{repo_id}", exist_ok=True)
        clone_repo(repo, f"{repo_playground}/{repo_id}")
    subprocess.run(['git', 'checkout', commit_id], cwd=repo_path, capture_output=True, text=True)
    structure = create_structure(repo_path)
    repo_info = {
        "repo": repo,
        "base_commit": commit_id,
        "structure": structure,
        "instance_id": instance_id,
    }
    return repo_info

def filter_out_test_files(files):
    """filter out test files from the repo"""
    return_f = []
    for item in files:
        if "test" not in item:
            return_f.append(item)
    return return_f

def filter_out_misc_files(files):
    """filter out misc files from the repo"""
    return_f = []
    for item in files:
        if "template" not in item:
            return_f.append(item)
    return return_f

def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}

    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure
