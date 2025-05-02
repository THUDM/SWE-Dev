#!/usr/bin/env python3
"""
Simple example showing how to use the configuration system
"""

import os
import sys
from pathlib import Path

# Ensure src is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config_value, print_config

def main():
    """Main function"""
    print("=== Configuration Example ===\n")
    
    # Print full config
    print("Full configuration:")
    print_config()
    
    # Access specific values directly
    print("\nAccessing specific values:")
    print(f"conda_bin = {get_config_value('paths.conda_bin')}")
    print(f"local_repo_dir = {get_config_value('paths.local_repo_dir')}")
    
    # Access a section
    print("\nAccessing Data Collection settings:")
    max_repos = get_config_value('data_collection.max_repos')
    max_pulls = get_config_value('data_collection.max_pulls')
    num_workers = get_config_value('data_collection.num_workers')
    
    print(f"max_repos = {max_repos}")
    print(f"max_pulls = {max_pulls}")
    print(f"num_workers = {num_workers}")
    
    # Example of using configuration in code
    print("\nExample usage in pipeline:")
    print(f"Starting data collection with {num_workers} workers...")
    print(f"Processing up to {max_repos} repositories and {max_pulls} pull requests per repo")
    
    # Example of using with default values
    timeout = get_config_value('test_evaluation.timeout', 300)
    print(f"Using timeout: {timeout} seconds")

if __name__ == "__main__":
    main() 