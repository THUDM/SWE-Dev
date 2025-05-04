
import ast
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

from swedev.utils.utils import clone_repo
from swedev.utils.preprocess import parse_python_file
from swedev.config import Config
from swedev.utils.preprocess import filter_none_python, filter_out_test_files

def has_python_files(path, max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return False
    try:
        for entry in path.iterdir():
            if entry.is_file() and entry.suffix == '.py':
                return True
            if entry.is_dir():
                if has_python_files(entry, max_depth, current_depth + 1):
                    return True
    except Exception as e:
        return False
    
    return False

def get_tree_string(directory, max_depth=3):
    result = []
    counts = {'dirs': 0, 'files': 0}
    def inner_tree(path, prefix="", depth=0):
        if depth >= max_depth:
            return
        valid_entries = []
        for entry in path.iterdir():
            if entry.is_file() and entry.suffix == '.py' and not "test" in entry.name:
                valid_entries.append(entry)
            elif entry.is_dir() and has_python_files(entry, max_depth, depth + 1):
                valid_entries.append(entry)
        
        valid_entries.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
        for i, entry in enumerate(valid_entries):
            is_last = i == len(valid_entries) - 1
            symbol = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")
            
            result.append(f"{prefix}{symbol}{entry.name}")
            
            if entry.is_dir():
                counts['dirs'] += 1
                inner_tree(entry, next_prefix, depth + 1)
            else:
                counts['files'] += 1
    
    root = Path(directory)
    if has_python_files(root, max_depth):
        result.append(root.name)
        inner_tree(root)
        result.append(f"\n{counts['dirs']} directories, {counts['files']} Python files")
    else:
        result.append(f"{root.name} (no Python files)")    
    return '\n'.join(result)

def parse_patch(patch_content: str) -> List[Tuple[str, int, int]]:
    file_ranges = []
    file_path = None
    for line in patch_content.splitlines():
        if line.startswith("diff --git"):
            match = re.search(r"diff --git a/(\S+) b/\1", line)
            if match:
                file_path = match.group(1)
        elif line.startswith("@@") and file_path:
            match = re.search(r"@@ -\d+,\d+ \+(\d+),(\d+) @@", line)
            if match:
                start_line = int(match.group(1))
                length = int(match.group(2))
                file_ranges.append((file_path, start_line, start_line + length - 1))
    return file_ranges

def get_code_block(node: ast.AST, lines: List[str]) -> str:
    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
        start = node.lineno - 1
        end = node.end_lineno
        return "".join(lines[start:end])
    return ""

def find_containing_blocks(file_path: str, start_line: int, end_line: int) -> str:
    with open(file_path, 'r') as f:
        source = f.read()
    lines = source.splitlines(keepends=True)
    blocks = []
    
    try:
        tree = ast.parse(source)
        
        def is_line_in_node(node, start, end):
            return (node.lineno <= start <= node.end_lineno or 
                    node.lineno <= end <= node.end_lineno or
                    (start <= node.lineno and end >= node.end_lineno))
        
        def find_blocks_in_node(node, lines):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if is_line_in_node(node, start_line, end_line):
                    blocks.append((node.lineno, get_code_block(node, lines)))
            
            for child in ast.iter_child_nodes(node):
                try:
                    find_blocks_in_node(child, lines)
                except:
                    continue
        
        find_blocks_in_node(tree, lines)
        sorted_blocks = [block for _, block in sorted(blocks, key=lambda x: x[0])]
        return "\n".join(sorted_blocks)
    
    except Exception as e:
        logging.critical(f'Error parsing file: {e}')
        return None

def get_location(data):
    """Process single instance."""
    structure = get_project_structure_from_scratch(
        data["repo"], data["base_commit"], data["instance_id"], Config.playground_path
    )
    if not structure:
        print('[No structure found]')
        return None
    instance_id = structure["instance_id"]

    structure = structure["structure"]
    filter_none_python(structure)
    filter_out_test_files(structure)

    # localize in file patches
    patch = data["patch"]
    file_ranges = parse_patch(patch)
    print(file_ranges)
    repo_name = data["repo"]
    commit_id = data["base_commit"]
    repo_id = f'{instance_id}_{repo_name.replace("/", "_")}_{commit_id}'
    repo_playground = os.path.join(Config.playground_path, repo_id, repo_name.split("/")[-1])

    try:
        subprocess.run(['git', 'add', '.'], cwd=repo_playground, capture_output=True, text=True)
        subprocess.run(['git', 'stash'], cwd=repo_playground, capture_output=True, text=True)
        subprocess.run(['git', 'stash', 'clear'], cwd=repo_playground, capture_output=True, text=True)
        subprocess.run(['git', 'checkout', commit_id], cwd=repo_playground, capture_output=True, text=True)
    except Exception as e:
        pass

    patch_blocks = []
    for file_path, start_line, end_line in file_ranges:
        if not file_path.endswith(".py"):
            continue
        try:
            code_block = find_containing_blocks(os.path.join(repo_playground, file_path), start_line, end_line)
            if code_block:
                patch_blocks.append({
                    "file": file_path,
                    "code": code_block
                })
        except Exception as e:
            logging.critical(f"Error localizing patch: {e}")
            pass
    project_tree = get_tree_string(repo_playground).strip()
    return {
        "patch_blocks": patch_blocks,
        "project_tree": project_tree
    }

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