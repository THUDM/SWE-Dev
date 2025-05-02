# 🚀 SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling

📝 [Blog](https://www.notion.so/ubecwang/1bc32cf963e080b2a01df2895f66021f?v=1bc32cf963e0810ca07e000c86c4c1e1) | 🤗 [Huggingface](https://huggingface.co/THUDM/SWE-Dev-32B) | 💻[Github](https://github.com/UbeCc/SWE-Dev)

This repository is a comprehensive pipeline for creating developer-oriented datasets from GitHub repositories, including issue tracking, code localization, test case generation, and evaluation.

## 🔄 Pipeline Overview

### Step 0: 🛠️ Configuration Setup

SWE-Dev uses [Hydra](https://hydra.cc/) for configuration management. All settings are stored in a single YAML file.

#### Configuration File

The main configuration file is located at `conf/config/default.yaml` and contains settings for all pipeline stages:

```yaml
# Basic directory settings
paths:
  conda_bin: /path/to/conda
  conda_base: /path/to/conda_dir
  local_repo_dir: /path/to/repos
  playground: /path/to/playground

# API credentials
github:
  tokens: []  # Add your GitHub tokens here

# OpenAI settings
openai:
  base_url: http://api.openai.com/v1
  base_model: gpt-4o
  api_key: ${oc.env:OPENAI_API_KEY,sk-test}

# Pipeline stage-specific model settings
# These settings allow using different models for each stage
localizer:
  model: ${openai.base_model}
  base_url: ${openai.base_url}

description:
  model: ${openai.base_model}
  base_url: ${openai.base_url}

testcase:
  model: ${openai.base_model}
  base_url: ${openai.base_url}
  revise_rounds: 0  # Number of rounds to revise testcases
  debug: false

# Other settings
data_collection:
  max_repos: 5000
  # ...

code_localization:
  # ...

test_generation:
  # ...
```

#### Validating Configuration

To validate your configuration:

```bash
python -m src.config --validate
```

#### Viewing Configuration

To view the current configuration:

```bash
python -m src.config --print
```

#### Overriding Configuration in Command Line

You can override any configuration value when running scripts:

```bash
# Override configuration values
python your_script.py paths.local_repo_dir=/new/path github.tokens=[token1,token2]
```

#### Using Configuration in Code

SWE-Dev provides a simple interface for accessing configuration values:

##### Option 1: Using the Config Class (Recommended)

```python
from src.config import Config

# Access basic configuration
conda_base = Config.conda_base
github_tokens = Config.github_tokens

# Access stage-specific settings
localizer_model = Config.Localizer.model
description_model = Config.Description.model
testcase_model = Config.Testcase.model
revise_rounds = Config.Testcase.revise_rounds

# Dynamic access for any config value
custom_setting = Config.get("data_collection.max_repos", 5000)

# Validate configuration
errors = Config.validate()
if errors:
    print("Configuration errors:", errors)
```

##### Option 2: Using get_config_value (Legacy)

```python
from src.config import get_config_value

# Access configuration values
conda_bin = get_config_value('paths.conda_bin')
max_repos = get_config_value('data_collection.max_repos', 5000)  # With default value
```

#### Environment Variables Fallbacks

Config values can be specified through environment variables as fallbacks. For example:
- `OPENAI_API_KEY` for OpenAI API access
- `GITHUB_TOKENS` for GitHub tokens (comma-separated)

Note: The configuration system will prioritize values in the YAML file over environment variables.

### Step 1: 📊 Data Collection from GitHub

Set up your configuration in `conf/config/default.yaml` with GitHub tokens and repository directories before running these commands.

#### Option 1: Collect Top PyPI Repositories
```bash
python -m src.issues.get_top_pypi \
    --max_repos 5000 \
    --output_folder results/issues/top_pypi \
    --num_workers 8 \
    --start_at 0
```
> ⚠️ Note: Keep concurrency lower to respect GitHub rate limits

#### Option 2: Custom GitHub Crawler
```bash
python -m src.issues.github_crawler \
    --start_page 0 \
    --end_page 2000 \
    --min_stars 500 \
    --max_repos 20000 \
    --workers 64 \
    --delay 10
```

#### Process the repositories
```bash
python -m src.issues.get_tasks_pipeline \
    --repo_file results/issues/top_pypi/gh-urls.jsonl \
    --output_folder results/issues \
    --cutoff_date 20210101 \
    --num_workers 64 \
    --max_pulls 1000
```

This will clone repositories to the directory specified by `local_repo_dir` in your configuration.

If you encounter persistent `404 - Error` messages, manually terminate and combine results:
```bash
python -m src.issues.get_tasks_pipeline \
    --repo_file results/issues/top_pypi/gh-urls-pr-3-star-5.jsonl \
    --output_folder results/issues \
    --combine_results
```

### Step 2: 🔍 Code Localization

Locate relevant files for the issues identified in Step 1:
```bash
python -m src.localizer.localize \
    --dataset results/issues/all_tasks.jsonl \
    --output_folder results/location \
    --output_file loc_outputs.jsonl \
    --top_n 5 \
    --num_workers 128
```

For parallel environments, create a base environment first to avoid Conda concurrent installation issues:
```bash
conda create -n swedevbase python=3.11 -y
conda create -n {env_name} --clone swedevbase
```

### Step 3: 📝 Generate Test Cases

First, generate descriptions:
```bash
python -m src.testcases.get_descriptions.py \
    --loc_file results/dataset_wo_description.jsonl \
    --top_n 5 \
    --output_folder results/descs-0227 \
    --num_workers 100
```

Then generate test cases:
```bash
python -m src.testcases.get_testcases \
    --loc_file results/descs-0227/output.jsonl \
    --top_n 5 \
    --output_folder results/testcases-0227/ \
    --num_workers 30
```

### Step 4: 🧪 Evaluate Test Cases

#### Docker Method
First, build a Docker image with required dependencies:
```bash
# Install comprehensive development environment
apt update && apt install -y \
    build-essential g++ gcc cmake make autoconf automake libtool pkg-config git curl wget unzip python3-dev \
    python3-pip python3-venv python-is-python3 libssl-dev libbz2-dev liblzma-dev zlib1g-dev libffi-dev \
    libsqlite3-dev libreadline-dev libgdbm-dev libdb-dev libexpat1-dev libxml2-dev \
    libxslt1-dev libyaml-dev libevent-dev libboost-all-dev libprotobuf-dev protobuf-compiler \
    libcurl4-openssl-dev libjpeg-dev libpng-dev libtiff-dev libfreetype-dev libx11-dev \
    libxext-dev libxrender-dev libxrandr-dev libxi-dev libxtst-dev libxinerama-dev libxkbcommon-dev libxkbcommon-x11-dev \
    libfontconfig1-dev libharfbuzz-dev libpango1.0-dev libcairo2-dev libgtk-3-dev libqt5widgets5t64 \
    qtbase5-dev qttools5-dev-tools libtbb-dev libopenblas-devliblapack-dev libatlas-base-dev \
    libsuitesparse-dev libeigen3-dev libgmp-dev libmpfr-dev libboost-python-dev libbz2-dev liblz4-dev \
    libzstd-dev libarchive-dev libsnappy-dev libuv1-dev librocksdb-dev libwebp-dev libxmlsec1-dev libgsl-dev \
    libgflags-dev libgoogle-glog-dev libhdf5-dev libtiff5-dev libyaml-cpp-dev libgd-dev default-jdk \
    openjdk-11-jdk openjdk-17-jdk maven gradle nodejs npm ruby-dev perl lua5.3 rustc cargo golang-go clang llvm lldb valgrind \
    ccache lcov doxygen graphviz gdb bison flex swig ninja-build libapache2-mod-php php-cli php-dev
```

Run the evaluation container:
```bash
docker run -d --network host \
  -v /raid:/raid \
  -w /raid/SWE-Dev \
  --restart always \
  testcase-image:latest \
  /raid/swedev/miniforge3/envs/swedev/bin/python -m src.testcases.eval_testcases \
  --dataset /raid/SWE-Dev/results/testcases-0218/output.jsonl \
  --output_folder /raid/SWE-Dev/results/evaluation-0218 \
  --num_workers 48
```

You should use **absolute path** when mounting directories

#### Non-Docker Method
```bash
python -m src.testcases.eval_testcases \
    --dataset results/testcases-0218/output.jsonl \
    --output_folder results/evaluation-0218 \
    --num_workers 32
```

### Step 5: 📈 View Evaluation Results

```bash
python -m src.testcases.eval_testcases \
    --dataset results/evaluation-0218/evaluated_testcases \
    --show_report
```

### Step 6: 📦 Create Final Dataset

```bash
python src/testcases/swebench_formatter.py \
    --dataset results/trajectory/qwen-45round-v0227.jsonl \
    --output_folder results/swedata \
    --output_name swe-qwen-45round-v0227.jsonl \
    --dataset_type openhands
```

## 🙏 Acknowledgements

We thank the following open-source projects for their contributions:

- [**SWE-bench**](https://github.com/SWE-bench/SWE-bench)

- [**Agentless**](https://github.com/OpenAutoCoder/Agentless)

- [**OpenHands**](https://github.com/All-Hands-AI/OpenHands)