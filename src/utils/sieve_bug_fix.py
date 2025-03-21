import concurrent.futures
import json
import random
from src.utils.utils import *

def get_url():
    urls = ["https://api.openai.com/v1"]
    return random.choice(urls)

def get_model():
    return "Llama-3.3-70B-Instruct"

def process_data(data):
    if 'is_debug' in data.keys():
        del data['is_debug']
    problem_statement = data.get("problem_statement", "").strip()
    
    if not problem_statement:
        return None

    prompt = (
        f"You are an AI assistant tasked with determining whether a given problem statement "
        f"describes a bug that needs to be fixed. A bug is an issue where something is not working "
        f"as intended, such as incorrect behavior, crashes, or errors. Also, you should filter out the statements that are on windows or macos. "
        f"Answer with 'yes' if the problem "
        f"statement clearly describes a bug, and 'no' otherwise. Only respond with 'yes' or 'no'.\n\n"
        f"Problem statement:\n{problem_statement}"
    )
    is_debug = 0
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response, _ = call(messages=[{"role": "user", "content": prompt}], model=get_model(), base_url=get_url())
            if response.strip().lower() in ['yes', 'no']:
                if response.strip().lower() == 'yes':
                    is_debug = 1
                else:
                    is_debug = -1
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")

        if attempt < max_retries - 1:
            print(f"Retrying with adjusted temperature for attempt {attempt + 2}...")
            try:
                response, _ = call(messages=[{"role": "user", "content": prompt}], temperature=0.7, model=get_model(), base_url=get_url())
                if response.strip().lower() in ['yes', 'no']:
                    if response.strip().lower() == 'yes':
                        is_debug = 1
                    else:
                        is_debug = -1
            except Exception as e:
                print(f"Retry {attempt + 2} failed with error: {e}")

    # by default, is_debug = 0
    data['is_debug'] = is_debug
    with open('debug_instances_true_v2.jsonl', 'a') as f:
        f.write(json.dumps(data) + '\n')

def process_jsonl(file_path):
    with open(file_path, 'r') as f:
        data_list = [json.loads(line) for line in f]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_data, data_list))    
    return results

if __name__ == "__main__":
    file_path = 'results/issues/all_tasks.jsonl'
    try:
        results = process_jsonl(file_path)
        print("Processing complete.")
        print(results)
    except Exception as e:
        print(f"An error occurred: {e}")