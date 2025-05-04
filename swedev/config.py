"""
Simple configuration module for SWE-Dev based on Hydra.
"""

import os
import sys
from typing import Any, List

import hydra
from omegaconf import DictConfig, OmegaConf
from functools import lru_cache

CONFIG_PATH = "../conf"

@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_main_get_config(cfg: DictConfig) -> DictConfig:
    """Hydra entry point to get configuration"""
    return cfg

# Cached config to prevent multiple Hydra initializations
@lru_cache()
def get_config() -> DictConfig:
    """
    Get configuration with caching to prevent multiple Hydra initializations
    """
    # Set Hydra to not change working directory and not print output
    os.environ["HYDRA_FULL_ERROR"] = "1"
    
    # Save original args to restore later
    argv = sys.argv.copy()
    
    # Temporarily modify sys.argv to contain only the script name
    sys.argv = [sys.argv[0]]
    
    try:
        with hydra.initialize_config_module(config_module="conf"):
            cfg = hydra.compose(config_name="config")
        return cfg
    finally:
        sys.argv = argv

def get_config_value(path: str, default: Any = None) -> Any:
    """
    Get a configuration value by path
    
    Args:
        path: Configuration path like "paths.conda_bin"
        default: Default value if not found
        
    Returns:
        The configuration value
    """
    cfg = get_config()
    parts = path.split('.')
    # Try to get value from config
    value = cfg
    try:
        for part in parts:
            value = value[part]
        return value
    except (KeyError, TypeError):
        # Try environment variable as fallback
        env_var = '_'.join(p.upper() for p in parts)
        env_value = os.environ.get(env_var)
        if env_value is not None:
            return env_value
        
        # If not found, return default
        return default

def validate_config() -> List[str]:
    """
    Validate the configuration and return error messages
    Returns empty list if valid
    """
    errors = []
    
    # Validate paths
    paths = ["paths.conda_bin", "paths.conda_base", "paths.local_repo_dir"]
    for path in paths:
        path_value = get_config_value(path)
        if path_value and not os.path.exists(path_value):
            errors.append(f"Path does not exist: {path} = {path_value}")
    
    github_tokens = get_config_value("github.tokens", [])
    if not github_tokens:
        env_tokens = os.environ.get("GITHUB_TOKENS", "")
        if env_tokens:
            github_tokens = env_tokens.split(",")
    
    if not github_tokens or (len(github_tokens) == 1 and not github_tokens[0]):
        errors.append("GitHub tokens not set. Required for API access.")
    
    return errors

# Load values from config for module-level constants
CONDA_BIN = get_config_value("paths.conda_bin")
CONDA_BASE = get_config_value("paths.conda_base")
LOCAL_REPO_DIR = get_config_value("paths.local_repo_dir")
PLAYGROUND_PATH = get_config_value("paths.playground")
GITHUB_TOKENS = get_config_value("github.tokens", [])
if not GITHUB_TOKENS and os.environ.get("GITHUB_TOKENS"):
    GITHUB_TOKENS = os.environ.get("GITHUB_TOKENS", "").split(",")
OPENAI_BASE_URL = get_config_value("openai.base_url")
OPENAI_BASE_MODEL = get_config_value("openai.base_model")
OPENAI_API_KEY = get_config_value("openai.api_key")
OPENAI_MAX_TOKENS = get_config_value("openai.max_tokens", 16384)
MODEL = OPENAI_BASE_MODEL

# Create a unified Config class for simplified imports
class Config:
    """
    Unified configuration class for easy imports.
    Usage: from src.config import Config
    """
    # Path settings
    conda_bin = CONDA_BIN
    conda_base = CONDA_BASE
    local_repo_dir = LOCAL_REPO_DIR
    playground_path = PLAYGROUND_PATH
    
    # Github settings
    github_tokens = GITHUB_TOKENS
    
    # OpenAI settings
    openai_base_url = OPENAI_BASE_URL
    openai_base_model = OPENAI_BASE_MODEL
    openai_api_key = OPENAI_API_KEY
    
    class Description:
        model = get_config_value("description.model", OPENAI_BASE_MODEL)
        base_url = get_config_value("description.base_url", OPENAI_BASE_URL)
        api_key = get_config_value("description.api_key", OPENAI_API_KEY)
        max_tokens = get_config_value("description.max_tokens", 16384)
    
    class Testcase:
        model = get_config_value("testcase.model", OPENAI_BASE_MODEL)
        base_url = get_config_value("testcase.base_url", OPENAI_BASE_URL)
        revise_rounds = get_config_value("testcase.revise_rounds", 0)
        max_tokens = get_config_value("testcase.max_tokens", 16384)
        api_key = get_config_value("testcase.api_key", OPENAI_API_KEY)
    
    @staticmethod
    def get(path: str, default: Any = None) -> Any:
        """Get any config value by path"""
        return get_config_value(path, default)
    
    @staticmethod
    def validate() -> List[str]:
        """Validate configuration"""
        return validate_config()

def print_config():
    """Print current configuration"""
    cfg = get_config()
    if cfg is not None:
        print(OmegaConf.to_yaml(cfg))
    else:
        print("Failed to load configuration")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SWE-Dev Configuration Tool")
    parser.add_argument('--print', action='store_true', help='Print current configuration')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    
    args = parser.parse_args()
    
    if args.print:
        print_config()
    elif args.validate:
        errors = validate_config()
        if errors:
            for error in errors:
                print(f"Error: {error}")
            sys.exit(1)
        else:
            print("Configuration is valid")
    else:
        print_config()