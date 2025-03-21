import argparse
import ast
import json
import os
import random
import time
import jsonlines

def extract_content_from_patch(patch: str) -> str:
    file_content = []
    is_content = False

    for line in patch.splitlines():
        if line.startswith('+++'):
            is_content = True
            continue  

        if is_content:
            if line.startswith('+') and not line.startswith('+++'):
                file_content.append(line[1:]) 
            elif line.startswith('-'): 
                continue
            elif line.startswith('@@'): 
                continue

    return "\n".join(file_content)

def extract_test_functions(source_code: str) -> list[str]:
    try:
        tree = ast.parse(source_code)
        test_functions = []
        class_name = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name            
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    if class_name:
                        test_functions.append(f"{node.name} ({class_name})")
                    else:
                        test_functions.append(node.name)
        return test_functions    
    except SyntaxError:
        return []

def parse_testcase(source_code: str, file_path: str) -> str:
    if not source_code.endswith("\n"):
        source_code += "\n\n"
    elif not source_code.endswith("\n\n"):
        source_code += "\n"
    
    patch = [
        f"diff --git a/{file_path} b/{file_path}",
        "new file mode 100644",
        "index 0000000..0000000",
        f"--- /dev/null",
        f"+++ b/{file_path}",
        "@@ -0,0 +1,{} @@".format(len(source_code.splitlines()))
    ]
    lines = source_code.splitlines()
    formatted_lines = [f"+{line.rstrip()}" for line in lines]
    patch.extend(formatted_lines)
    return "\n".join(patch)

def parse_testcase_with_functions(source_code: str, file_path: str) -> tuple[str, list[str]]:
    patch = parse_testcase(source_code, file_path)
    return patch 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts json swebench format')
    parser.add_argument('--dataset', type=str, help='Input files (comma-separated)')
    # swe keys: [repo, instance_id, base_commit, patch, test_patch, problem_statement, 
    #           hints_text, created_at, version, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit]
    parser.add_argument('--output_folder', type=str, default="results", help='Output folder')
    parser.add_argument('--output_name', type=str, default="swedev-gen.jsonl", help='Output file')
    parser.add_argument('--dataset_type', type=str, default='default')
    args = parser.parse_args()

    # Create output file path
    output_file = os.path.join(args.output_folder, args.output_name)
    dataset_files = [file.strip() for file in args.dataset.split(',')]

    all_data = []
    for file in dataset_files:
        if file.endswith(".json"):
            with open(file, 'r') as f:
                all_data.extend(json.load(f))
        elif file.endswith(".jsonl"):
            with jsonlines.open(file, 'r') as reader:
                all_data.extend(list(reader))
        else:
            raise ValueError(f"Unsupported file format: {file}")

    results = []

    if args.dataset_type == 'default':
        for data in all_data:
            repo = data["repo"]
            instance_id = data["instance_id"]
            base_commit = data["base_commit"]
            patch = data["patch"]
            problem_statement = data["problem_statement"]
            hints_text = data["hints_text"] if "hints_text" in data else ""
            created_at = data["created_at"] if "created_at" in data else "2025-01-23T23:59:59"
            version = "0.0"
            FAIL_TO_PASS = []
            PASS_TO_PASS = []
            patches = []
            
            for i, test in enumerate(data['test']):
                try:
                    test_file = f'swedev_test_{i}.py'
                    if test['content']:
                        patches.append(parse_testcase(test['content'], test_file))
                    for function in test['FAIL_TO_PASS']:
                        # if function['status'] == 'normal':
                        FAIL_TO_PASS.append(function['name'].replace("swedev_test.py", test_file))
                except Exception as e:
                    print(e)
            
            if not FAIL_TO_PASS:
                continue
            
            assert FAIL_TO_PASS and patches, "Empty FAIL_TO_PASS or patches"
            
            results.append({
                "repo": repo,
                "instance_id": instance_id,
                "base_commit": base_commit,
                "patch": patch,
                "test_patch": "\n".join(patches) + "\n",
                "problem_statement": problem_statement,
                "hints_text": hints_text,
                "created_at": created_at,
                "version": version,
                "FAIL_TO_PASS": FAIL_TO_PASS,
                "PASS_TO_PASS": PASS_TO_PASS,
                "environment_setup_commit": "",
                "description": ""
            })

        random.shuffle(results)
        results = list({item['instance_id']: item for item in results}.values())

    elif args.dataset_type == 'openhands':
        for data in all_data:
            result = data['instance']
            result['solution'] = data['test_result']['git_patch']
            if not "environment_setup_commands" in result.keys():
                print('Old version data detected')
                continue
            result['tests'] = [{
                "content": extract_content_from_patch(result['test_patch']),
                "env": result['environment_setup_commands'],
                "id": 0,
            }]
            if not result['description']:
                result['description'] = ""
            results.append(result)
    else:
        raise NotImplementedError

    os.makedirs(args.output_folder, exist_ok=True)
    print(f'Total: {len(results)} pieces.')
    print(f'Output file is: {output_file}')
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(results)