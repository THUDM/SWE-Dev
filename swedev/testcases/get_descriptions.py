import argparse
import json
import logging
import os
import re
import jsonlines
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from swedev.utils.prompts import *
from swedev.utils.utils import *
from tqdm import tqdm
from swedev.config import Config

call_counter = tqdm(desc="API Calls", unit="calls")
total_counter = tqdm(desc="Progress", unit="items")
saved_counter = tqdm(desc="Saved", unit="items")

# Simple RPS tracking variables
request_count = 0
start_time = time.time()
rps_lock = threading.Lock()

def process_instance(instance, function_id, model, base_url, logger, writer):
    """Process a single test instance to generate description and save it"""
    global request_count
    
    # Track RPS
    with rps_lock:
        request_count += 1
    call_counter.update(1)
    
    result = call(
        messages=[{"role": "user", "content": SUMMARIZE_GHERKIN_TEST.format(instance["repo"], instance["problem_statement"], instance["patch"], instance["hints_text"])}],
        max_tokens=2048,
        model=model,
        base_url=base_url,
        logger=logger
    )
    if result == "Error":
        print("Too long when generating raw desc")
        result = call(
            messages=[{"role": "user", "content": SUMMARIZE_GHERKIN_TEST.format(instance["repo"], instance["problem_statement"], instance["patch"], "No Hints Text Provided")}],
            max_tokens=2048,
            model=model,
            base_url=base_url,
            logger=logger
        )     
    
    desc = call(
        messages=[{"role": "user", "content": MAKE_GHERKIN_TEST.format(instance["repo"], instance["problem_statement"], instance["patch"], instance["hints_text"], result)}],
        max_tokens=2048,
        model=model,
        base_url=base_url,
        logger=logger
    )
    if desc == "Error":
        print("Too long when generating desc")
        desc = call(
            messages=[{"role": "user", "content": MAKE_GHERKIN_TEST.format(instance["repo"], instance["problem_statement"], instance["patch"], "No Hints Text Provided", result)}],
            max_tokens=2048,
            model=model,
            base_url=base_url,
            logger=logger
        )
    pattern = r'```(?:gherkin\n|\n)(.*?)\n```'
    descs = re.findall(pattern, desc, re.DOTALL)

    total_counter.update(1)
    
    result_obj = {
        "repo": instance["repo"],
        "instance_id": instance["instance_id"],
        "problem_statement": instance["problem_statement"],
        "patch": instance["patch"],
        "created_at": instance["created_at"],
        "hints_text": instance["hints_text"],
        "base_commit": instance["base_commit"],
        "descs": descs,
        "model": model
    }
    
    with rps_lock:
        writer.write(result_obj)
        writer.flush()
        saved_counter.update(1)
    
    return result_obj

def generate_descriptions(args: argparse.Namespace) -> None:
    """
    Get test case descriptions for a list of test instances
    """
    logger = logging.getLogger(__name__)
    model = Config.Description.model
    base_url = Config.Description.base_url
    
    if args.dataset_file.endswith(".jsonl"):
        with jsonlines.open(args.dataset_file, "r") as f:
            test_instances = [line for line in f]
    else:
        with open(args.dataset_file, "r") as f:
            test_instances = json.load(f)
    
    total_test_instances = len(test_instances)
    print(f"Getting test case descriptions for {total_test_instances} test instances")
    
    total_counter.total = total_test_instances
    total_counter.refresh()
    
    def report_rps():
        global request_count, start_time
        while True:
            time.sleep(5) 
            with rps_lock:
                elapsed = time.time() - start_time
                rps = request_count / elapsed if elapsed > 0 else 0
                print(f"Current RPS: {rps:.2f}")
    
    rps_thread = threading.Thread(target=report_rps, daemon=True)
    rps_thread.start()
    
    with jsonlines.open(args.output_file, 'w') as writer:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for function_id, instance in test_instances.items():
                future = executor.submit(
                    process_instance, 
                    instance, 
                    function_id, 
                    model, 
                    base_url, 
                    logger,
                    writer
                )
                futures.append(future)
                
            for future in tqdm(futures, total=total_test_instances, desc="Processing", unit="items"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing instance: {e}")
                    traceback.print_exc()
    
    elapsed = time.time() - start_time
    final_rps = request_count / elapsed if elapsed > 0 else 0
    print(f"Final RPS: {final_rps:.2f}")
    print(f"Saved {saved_counter.n} descriptions to {args.output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    args.output_file = os.path.join(args.output_folder, "output.jsonl")
    generate_descriptions(args)

if __name__ == "__main__":
    main()