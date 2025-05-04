import argparse
import ast
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import traceback
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jsonlines
from swebench.collect.get_repo_structure import get_repo_structure
from tqdm import tqdm

from swedev.config import Config
from swedev.localizer.file_localizer import LLMFileLocalizer
from swedev.localizer.get_repo_structure import get_project_structure_from_scratch, get_files_and_classes_for_repos
from swedev.utils.preprocess import filter_none_python, filter_out_test_files

DEBUG = True

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

# begin of patch localization
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
            logging.critical(f"Error: {e}")
            pass
    project_tree = get_tree_string(repo_playground).strip()
    return {
        "patch_blocks": patch_blocks,
        "project_tree": project_tree
    }

def process_single_instance(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process single instance."""
    structure = get_project_structure_from_scratch(
        data["repo"], data["base_commit"], data["instance_id"], Config.playground_path
    )
    if not structure:
        print('[No structure found]')
        return None
    instance_id = structure["instance_id"]

    hints_text = data["hints_text"]
    problem_statement = data["problem_statement"]
    structure = structure["structure"]
    filter_none_python(structure)
    
    if not instance_id.startswith("pytest"):
        filter_out_test_files(structure)

    # localize in file patches
    patch = data["patch"]
    file_ranges = parse_patch(patch)
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
            else:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                raw_code = "".join(lines[start_line-1:end_line])
                patch_blocks.append({
                    "file": file_path,
                    "code": raw_code
                })
        except:
            pass
    project_tree = get_tree_string(repo_playground).strip()
    return {
        "repo": repo_name,
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "hints_text": hints_text,
        "patch": patch,
        "base_commit": commit_id,
        "created_at": data["created_at"],
        "hints_text": data["hints_text"],
        "patch_blocks": patch_blocks,
        "project_tree": project_tree,
        "test_patch": data["test_patch"],
    }

def localize(args: argparse.Namespace):
    dataset = args.dataset
    print("start loading files")
    if dataset.endswith('.json'):
        with open(args.dataset, 'r') as f:
            dataset = json.load(f)
    elif dataset.endswith('.jsonl'):
        with open(args.dataset, 'r') as f:
            dataset = [json.loads(line) for line in f]
    else:
        raise ValueError("Dataset file must be in JSON or JSONL format")
    print(f"end loading files, total: {len(dataset)} pieces.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_single_instance, data)
            for data in dataset
        ]
        
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    with open(args.output_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"Error processing localization: {str(e)}")
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    args.output_file = os.path.join(args.output_folder, args.output_file)
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    if os.path.exists(args.output_file):
        logging.warning("Output file already exists, will overwrite it.")
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        filename=f"{args.output_folder}/localize.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    localize(args)

if __name__ == "__main__":
    main()