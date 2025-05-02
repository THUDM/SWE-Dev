import json
import os

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, HfFolder, Repository
from src.config import get_config_value


def download():
    if not os.path.exists("datasets/"):
        os.mkdir("datasets/")

    dataset = load_dataset("princeton-nlp/SWE-bench")

    with open("datasets/swebench-train.json", "w") as f:
        json.dump([d for d in dataset["train"]], f, indent=2)
    with open("datasets/swebench-dev.json", "w") as f:
        json.dump([d for d in dataset["dev"]], f, indent=2)        
    with open("datasets/swebench-test.json", "w") as f:
        json.dump([d for d in dataset["test"]], f, indent=2)

def upload_to_huggingface(dataset_name, file_split_mapping, token):
    """
    Uploads multiple dataset files to the Hugging Face Hub as splits under dataset_name.
    
    Args:
        dataset_name (str): The Hugging Face dataset repository name (e.g., "username/dataset_name").
        file_split_mapping (dict): A dictionary mapping split names (e.g., "train", "test") to file paths.
        token (str): Hugging Face API token.
    """
    # Save the token for authentication
    HfFolder.save_token(token)
    api = HfApi()

    # Authenticate and check user
    user = api.whoami(token=token)
    print(f"Authenticated as: {user['name']}")

    # Check if the dataset exists on the Hugging Face Hub
    try:
        api.dataset_info(dataset_name, token=token)
        print(f"Dataset {dataset_name} already exists on Hugging Face Hub.")
    except Exception:
        print(f"Dataset {dataset_name} does not exist. Creating a new repository...")
        api.create_repo(repo_id=dataset_name, repo_type="dataset", token=token)

    # Ensure all files exist
    for split, file_path in file_split_mapping.items():
        if not os.path.exists(file_path):
            raise ValueError(f"File for split '{split}' does not exist: {file_path}")

    # Load each file into a DatasetDict
    dataset_dict = {}
    for split, file_path in file_split_mapping.items():
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"Loading file '{file_path}' for split '{split}' (format: {file_ext})")

        if file_ext == ".csv":
            dataset = Dataset.from_csv(file_path)
        elif file_ext == ".json":
            dataset = Dataset.from_json(file_path)
        elif file_ext == ".jsonl":
            dataset = Dataset.from_json(file_path, split="train")
        else:
            raise ValueError(f"Unsupported file format for split '{split}': {file_ext}. Only .csv, .json, and .jsonl are supported.")

        dataset_dict[split] = dataset

    # Convert to DatasetDict
    dataset_dict = DatasetDict(dataset_dict)

    # Push the entire DatasetDict to the Hugging Face Hub
    print(f"Uploading dataset to Hugging Face Hub under {dataset_name}...")
    dataset_dict.push_to_hub(dataset_name, token=token)
    print(f"Dataset uploaded successfully to {dataset_name}!")

def upload():
    DATASET_NAME = "SWE-Dev/SWE-Dev"
    TOKEN = get_config_value("huggingface.token", os.environ.get("HF_TOKEN"))
    if not TOKEN:
        raise ValueError("HF_TOKEN not configured. Please configure huggingface.token in your config file or set the HF_TOKEN environment variable.")
    FILE_SPLIT_MAPPING = {
        "train": "results/swedev.jsonl",
    }
    upload_to_huggingface(DATASET_NAME, FILE_SPLIT_MAPPING, TOKEN)
    
upload()
