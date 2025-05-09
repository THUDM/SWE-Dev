import json
import jsonlines
from typing import Dict, Any, List, Optional
import re
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import os
import time
import argparse

def get_llm(prompt: str, api_key: str, base_url: str) -> Optional[str]:
    """Call LLM API with basic retry logic"""
    for attempt in range(3):
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="glm-4-flash",
                temperature=0.7,
                top_p=0.8,
                stream=False,
                max_tokens=1024
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call attempt {attempt+1}/3 failed: {str(e)}")
            if attempt < 2:
                time.sleep(2)
    
    return None

def create_vague_problem_statement_prompt(instance):
    prompt = f"""
    Evaluate if this problem statement is extremely vague based on these criteria:
    1. Completely missing any specifics about what to implement/fix
    2. Severely lacking any context or background information
    3. No clear requirements at all
    4. Entirely undefined scope/boundaries
    5. No technical specifications whatsoever
    6. Provides no actionable information for a developer
    7. Contains major contradictions or inconsistencies

    Problem Statement: {instance["problem_statement"]}

    Return your judgment as: <judgement>True</judgement> ONLY if extremely vague and completely unusable, or <judgement>False</judgement> otherwise.
    Be very lenient in your evaluation - if the problem statement provides ANY useful information that could help a developer start working, consider it acceptable and return False.
    Only mark as True if the problem statement is so vague that it would be completely impossible to work with.
    """
    return prompt

def is_vague_problem_statement(instance, api_key, base_url):
    prompt = create_vague_problem_statement_prompt(instance)
    response = get_llm(prompt, api_key, base_url)
    
    if not response:
        return False  # Consider not vague if API fails (less strict)
        
    try:
        match = re.search(r'<judgement>(True|False)</judgement>', response, re.IGNORECASE)
        if match:
            result = match.group(1).lower()
            return result == 'true'

        if 'true' in response.lower() and 'false' not in response.lower():
            return True
        elif 'false' in response.lower() and 'true' not in response.lower():
            return False

        print(f"Warning: Unclear response from LLM: {response}")
        return False
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return False

def check_code_quality(instance, min_patch_length=50, max_patch_length=200000):
    """Check basic code quality based on heuristics, with more lenient thresholds"""
    patch = instance.get("patch", "")
    
    # Check patch length (more lenient)
    if len(patch) < min_patch_length:
        return False, "Patch too short"
    
    if len(patch) > max_patch_length:
        return False, "Patch too long"
    
    # Check for test files only
    test_files_only = True
    file_pattern = re.compile(r"diff --git a/(.*?) b/")
    file_matches = file_pattern.findall(patch)
    
    if not file_matches:
        return False, "No files found in patch"
    
    for file in file_matches:
        if not (file.endswith("test.py") or file.endswith("tests.py") or 
                "test/" in file or "spec/" in file or file.endswith("_test.go")):
            test_files_only = False
            break
    
    if test_files_only and len(file_matches) > 0:
        return False, "Patch contains only test files"
    
    # Check for meaningful changes (not just comments or whitespace) - more lenient
    content_lines = 0
    added_lines = re.findall(r'\n\+[^\+]', patch)
    for line in added_lines:
        stripped = line.replace('\n+', '').strip()
        if stripped and not stripped.startswith('//') and not stripped.startswith('#'):
            content_lines += 1
    
    if content_lines < 3:  # Reduced from 5 to 3
        return False, "Too few meaningful added lines"
    
    return True, "Passed code quality checks"

def check_problem_quality(instance, disallowed_phrases):
    """Check problem statement quality based on heuristics"""
    problem = instance.get("problem_statement", "")
    
    # Check length - more lenient
    if len(problem) < 30:  # Reduced from 50 to 30
        return False, "Problem statement too short"
    
    if len(problem) > 3000:  # Increased from 2000 to 3000
        return False, "Problem statement too long"
    
    # Check for disallowed phrases
    lower_problem = problem.lower()
    for phrase in disallowed_phrases:
        if phrase.lower() in lower_problem:
            return False, f"Problem contains banned phrase: {phrase}"
    
    return True, "Passed problem quality checks"

def process_instance(instance, api_key, base_url):
    try:
        # Check if required fields exist
        for field in ["problem_statement", "patch"]:
            if field not in instance or not instance[field]:
                return None
        
        # Check if ci_name_list is empty
        if "ci_name_list" not in instance or not instance["ci_name_list"] or len(instance["ci_name_list"]) <= 3:
            return None
        
        # Step 1: Files that have more than 5 diffs
        count = instance["patch"].count("diff --git a/")
        if count > 5:
            return None
        
        # Step 2: Check basic code quality
        code_passed, code_reason = check_code_quality(instance)
        if not code_passed:
            return None
        
        # Step 3: Check problem quality using custom heuristics
        disallowed_phrases = []  # Add any disallowed phrases here if needed
        problem_passed, problem_reason = check_problem_quality(instance, disallowed_phrases)
        if not problem_passed:
            return None
        
        # Step 4: Check if the problem statement is vague (using LLM)
        if is_vague_problem_statement(instance, api_key, base_url):
            return None
        
        return instance
    except Exception as e:
        print(f"Error processing instance: {str(e)}")
        return None

def process_data_parallel(instances, output_file, api_key, base_url, num_workers=50):
    results = []
    processed_count = 0
    rejected_count = 0
    
    rejection_stats = {
        "total_rejected": 0,
        "reasons": {
            "too_many_diffs": 0,
            "poor_problem_quality": 0,
            "poor_code_quality": 0,
            "vague_problem": 0,
            "empty_ci_name_list": 0,
            "missing_required_fields": 0,
            "processing_error": 0
        }
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_instance, instance, api_key, base_url)
            for instance in instances
        ]

        with tqdm(total=len(futures), desc="Processing instances") as progress:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        with open(output_file, "a") as f:
                            f.write(json.dumps(result) + "\n")
                        processed_count += 1
                    else:
                        rejected_count += 1
                except Exception as e:
                    print(f"Error in processing: {str(e)}")
                    rejected_count += 1
                    rejection_stats["reasons"]["processing_error"] += 1
                progress.update(1)

    total = len(instances)
    acceptance_rate = (processed_count / total) * 100 if total > 0 else 0
    
    print(f"\nFiltering Results:")
    print(f"Processed {processed_count} instances successfully out of {total}")
    print(f"Rejected {rejected_count} instances")
    print(f"Acceptance rate: {acceptance_rate:.2f}%")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter code problem statements")
    parser.add_argument("--input", type=str, help="Input file path (.json or .jsonl)")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--api-key", type=str, help="API key for LLM service")
    parser.add_argument("--base-url", type=str, help="Base URL for LLM service")
    parser.add_argument("--workers", type=int, default=64, help="Number of worker threads")
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    api_key = args.api_key
    base_url = args.base_url
    num_workers = args.workers

    if os.path.exists(output_file):
        os.remove(output_file)

    if input_file.endswith(".json"):
        with open(input_file, "r") as f:
            instances = json.load(f)
    else:
        with jsonlines.open(input_file, "r") as f:
            instances = list(f)

    process_data_parallel(instances, output_file, api_key, base_url, num_workers)
    

import os

INPUT_FILE = "results/issues/all_tasks_with_test_2.jsonl"
# results/issues/all_tasks_with_test_2.jsonl
OUTPUT_FILE = "filtered_test.jsonl"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File '{INPUT_FILE}' not found.")
        return

    line_sizes = []  # Stores the size of each line
    filtered_lines = []  # Stores lines <= 1MB

    # Read the file and process line sizes
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            size = len(line.encode("utf-8"))  # Calculate size in bytes
            line_sizes.append(size)
            if size <= 1024 * 1024:  # Filter lines <= 1MB
                filtered_lines.append(line)

    # Save filtered lines to a new file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)

    print(f"Total lines: {len(line_sizes)}")
    print(f"Lines <= 1MB: {len(filtered_lines)}")
    print(f"Filtered lines saved to '{OUTPUT_FILE}'.")

    # If no lines are <= 1MB, stop further processing
    if not filtered_lines:
        print("No lines <= 1MB found.")
        return

if __name__ == "__main__":
    main()