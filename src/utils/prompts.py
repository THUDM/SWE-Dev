# Demo for gherkin description:

# Feature: Quiet mode in SQLFluff CLI

#   Scenario: Run sqlfluff fix with --quiet option
#     Given I have a SQL file with linting violations
#     When I run `sqlfluff fix` with the `--quiet` option
#     Then the output should only show the fix status and the number of fixes applied
#     And the output should not contain detailed information about each fix

#   Scenario: Run sqlfluff fix with --force and --quiet options
#     Given I have a SQL file with multiple linting violations
#     When I run `sqlfluff fix` with the `--force` and `--quiet` options
#     Then the output should only show the fix status and the number of fixes applied
#     And all fixes should be applied automatically

#   Scenario: Run sqlfluff fix with both --quiet and --verbose options
#     Given I have a SQL file with linting violations
#     When I run `sqlfluff fix` with both `--quiet` and `--verbose` options
#     Then I should see an error message stating that --quiet and --verbose cannot be used together
#     And the process should exit with an error code

SUMMARIZE_GHERKIN_TEST = """
You are a skilled test engineer. Your mission is to create a minimal, edge-case test scenario that serves to rigorously validate the effectiveness of the patch. This test case must satisfy the following conditions:
	1.	Fail with the unpatched code: Demonstrate the specific bug, issue, or limitation that the patch is designed to address. Ensure the test triggers this behavior reliably and consistently.
	2.	Pass with the patched code: Confirm that the patch resolves the issue without introducing new problems or regressions.

Focus on crafting a concise yet challenging input or situation that isolates the problem the patch addresses. Avoid superficial or trivial cases; instead, target scenarios that:
	- Exercise uncommon or edge-case code paths.
	- Test for boundary conditions or unexpected input.
	- Mimic realistic usage scenarios where the original behavior breaks.

- Repository name: {}
- GitHub issue description: {}
- Correction patch: 
```
{}
```
- Hints Text: {}

The generation is split into several steps. Now, you task is to write a testcase description for further generation. The description should reflect the modification of the patch. 
You should briefly analyse the problem description and hints text, finding out **where should be fixed**. After that, **give the description for testcase to be generated**. Do not give any unrelated greeting words!
"""

MAKE_GHERKIN_TEST = """
You are an experienced test engineer. Now I need you to write a test following the Gherkin syntax based on the information below. This test is used to verify whether the correction patch in the repository correctly solves the problem.
Please note that our goal is to make the source code fail the tests without correct patch, and pass the tests with correct patch.

- Repository name: {}
- GitHub issue description: {}
- Correction patch: 
```
{}
```
- Hints Text: {}

Here is the analysis for the testcases that you can refer to.
{}

Requirements:
1. Use the `Given-When-Then` structure of Gherkin.
2. Clearly describe the preconditions, triggering events, and expected results.
3. Ensure the test logic is clear and covers all relevant scenarios.

Please provide the Gherkin syntax test in the most concise way. Do not include any unrelated greetings! Do not give unimportant testcases like modificaton of README.
And you should wrap each of your gherkin test description with triple backticks. For example, ```gherkin\n{{YOUR DESCRIPTION}}\n```
"""

TESTCASE_GENERATION = """
You are a test engineer. Given a github issue description and the golden patch, your task is to build testcases that **reproduce the error** according to the patch. In detail, the testcases should reproduce the error in the issue description.

Your testcase will running at **root** of the project, please be care of the relative path to avoid path-related errors.

# Information provided

- **Repository name**: {}

- **GitHub issue description**: {}

- **Hints Text**: {}

- **Correction patch**: 
```
{}
```

- **Project tree (file depth less than 3)**: 
{}

- **Testcase Description**: {}

- **Relevant code segments in the original version**: 
```
{}
```

# Steps to follow:

1. **Identify the incorrect code**: Analyze the provided information to locate the error that the patch addresses. You should figure out what packages you need and what kind of testcases you should write.
2. **Generate the test case**: Write testcases that will **fail without the correction patch** and **pass with the correction patch**. Each testcase must be enclosed within `<testcase></testcase>` tags.
Please note *no additional execution will be done except for your testcase* You should make any segments in the code like create a new file or turn on an system variable.
Please note again that your testcase will running at root of the project, please be care of the relative path to avoid path-related errors.

# Format Requirements:

- Testcase: 
  - Wrap each test case in `<testcase></testcase>` tags.
  - Use triple backticks (```) to enclose the test code within the `<testcase></testcase>` tags.
  - The test cases must be ready to run with `pytest` and should include any necessary mock data or fixtures.

# Environment Information

- Python Version: 3.9
- Platform: Ubuntu 22.04.5 LTS
- Execution Command: python -m pytest --no-header -rA -p no:cacheprovider -W ignore::DeprecationWarning --continue-on-collection-errors --tb=short
- Execution Path: root directory of the project

# Example Solution

In `src/utils/csv_utils.py`:
```
from CSVconverter.src.utils import csv
def read_csv_and_sum(filename):
    \"\"\"Calculate the sum of all numbers in a CSV file\"\"\"
    total = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            total += row[0]
    return total
```
The code directly adds row[0] to total without validating if row[0] is an integer. If the CSV file contains non-numeric values (e.g., strings or empty fields), it will raise runtime errors like TypeError or ValueError. These errors matche the problem statement. So I'll write testcases here.

Fix Explanation:
	1.	It tries to convert row[0] to an integer using int().
	2.	If row[0] is not a valid integer, it skips that row using a try...except block.

The goal is to write test cases that:
	1.	Test case with non-numeric data in the CSV (should raise an error in the original code).
	2.	Same test case should now correctly handle non-numeric rows and calculate the sum of valid numeric values.

<testcase>
```python
import os
import pytest
from src.utils.csv_utils import read_csv_and_sum

@pytest.fixture
def create_csv_file():
    \"\"\"Fixture to create a temporary CSV file for testing.\"\"\"
    def _create_file(contents, filename="test.csv"):
        with open(filename, 'w') as f:
            f.write(contents)
        return filename
    yield _create_file
    # Cleanup after test
    if os.path.exists("test.csv"):
        os.remove("test.csv")

def test_valid_csv(create_csv_file):
    \"\"\"Test case with valid numeric data.\"\"\"
    filename = create_csv_file("1\\n2\\n3\\n")
    result = read_csv_and_sum(filename)
    assert result == 6  # Expected sum of numbers

def test_non_numeric_csv(create_csv_file):
    \"\"\"Test case with non-numeric data.\"\"\"
    filename = create_csv_file("1\\nabc\\n3\\n")
    with pytest.raises(TypeError):
        read_csv_and_sum(filename)

def test_empty_csv(create_csv_file):
    \"\"\"Test case with an empty CSV file.\"\"\"
    filename = create_csv_file("")
    result = read_csv_and_sum(filename)
    assert result == 0  # Expected sum is 0
```
</testcase>

# Final instructions

- **Test case format**: Ensure the tests follow `pytest` conventions and are ready to run and you should never enable dangerous commands like `ifconfig` and `iptables` in the testcase and setup commands.
- **Import files correctly**: You should carefully deal with functions and classes in the current package.
- **Patch validation**: The test case should fail when run against the unpatched code and pass after the patch is applied.
- **Be careful about other files related**: Your testcase may need to write or read files. You should make sure the files exist. Instructive path like `/path/to/dest` should be substitute to real path.
"""

REVISION_BEFORE_PROMPT = """
# Task Definition:
You are tasked with generating **improved test cases** because the previous test cases **passed without the patch**. Your goal is to maintain the original intention of the test cases, while ensuring that:
1. They **address the specific failures** shown in the error history.
2. They **fail before the patch is applied** (i.e., they should expose the original issue).
3. They **pass after applying the patch** (i.e., they should verify that the patch fixes the issue).

# Provided Information:

- **Repository name**: 
`{}`

- **GitHub issue description**: 
{}

- **Correction patch** (this patch passed with the previous test cases):
```
{}
```

- **Project tree (file depth less than 3)**: 
{}

- **Relevant code segments**: 
```
{}
```

### Given Test Cases
{}

### Task Instructions:
- **Preserve the original intent**: Ensure the new test cases still target the original issues that the patch is designed to fix.
- **Analyze the error history**: Review the error history to understand why the previous test cases passed without the patch.
- **Generate new test cases**: Write new test cases that expose the issue in the unpatched code and pass after applying the patch.
- **Format Requirements**: Your test cases should follow the original format. Specifically:
  - Each test case must be wrapped in `<testcase></testcase>`.
  - The test code should be enclosed in triple backticks (```) inside the `<testcase>`.
  - Any required Python packages should be listed within `<env></env>` tags.

### Example format:
<testcase>
```python
# Your improved test case here
```
</testcase>

<env>
# Required Python packages here
</env>

Remember, the new test cases must still fail on the unpatched code and **pass** after applying the patch. Strictly follow the format and preserve the original test intent!
"""

REVISION_AFTER_PROMPT = """
You are tasked with generating testcases for a given github issue.
The code with golden patch should pass the testcase while fail without golden patch. 
Now, the testcase **failed** even **after** applying the patch. You should improve it.

# Provided Information:

- **Repository name**: {}

- **GitHub issue description**: {}

- **Hints Text**: {}

- **Golden patch** (this patch passed with the previous test cases):
```
{}
```

- **Project Tree (file depth less than 3)**: 
{}

- **Relevant Code Segments**: 
```
{}
```

- **Available Relevant APIs**
```
{}
```

- **Wrong Testcase**
{}

- **Error History**
{}

# Task Instructions:
- **Analyze the error history carefully**: Review the error history to understand why the previous test cases passed without the patch. For example,
    - You should rewrite wrong testcases if error occurs on specific tests.
    - You should consider `import` dependencies when ImportError or errors like that occur.
- **Preserve the original intent**: Ensure the new test cases still target the original issues that the patch is designed to fix.
- **Format Requirements**: Your testcase should strictly follow the original format. Specifically:
  - Setup command should be wrapped in `<env></env>` tags. And the commands should be enclosed in triple backticks (```) inside the `<env>`.
  - Test case must be wrapped in `<testcase></testcase>` and the test code should be enclosed in triple backticks (```) inside the `<testcase>`.

# Example format:
<testcase>
```python
# Your improved test case here
```
</testcase>

<env>
```bash
# Required setup commands here
```
</env>

Remember, the new testcase must still fail on the unpatched code and **pass** after applying the patch. Strictly follow the format and preserve the original test intent!
"""

TESTCASE_FORMAT = """<testcase>
```python
{}
```
</testcase>

<env>
```bash
{}
```
</env>
"""

EXTRACT_API_PROMPT = """
Here is an error message, and you are required to extract the API signature or class name that raise the error.
You should strictly follow the format instruction, and do not include any unrelated greeting words.
The api should **directly** raise the error, and you should not include any other api that is not related to the error.
For safety, you should never use `os.system` in the testcase code. If you want to operate on system, you should use the commands in setup commands!

# Format Instruction
You should wrap the API signature or class name in the following format:

- If the error message is related to a function, the API signature should be in the following format:
```
<function>module1.module2.function_name(parameters)</function>
```
- If the error message is related to a class, the class name should be in the following format:
```
<class>module1.module2.class_name</class>
```
- If no API signature or class name is found, you should provide an empty string.
```
<empty></empty>
```

# Example
## Error Message
```
"output": {{"stdout": "============================= test session starts ==============================\ncollecting ... collected 0 items / 1 error\n\n==================================== ERRORS ====================================\n__________________________ ERROR collecting test_0.py __________________________\nImportError while importing test module '/mnt/nvme/playground/matze__pkgconfig-70_matze_pkgconfig_480e92f4c34dd0bc1ef243f23cdd98c1f2905ac6/pkgconfig/test_0.py'.\nHint: make sure your test modules/packages have valid Python names.\nTraceback:\n../../../miniforge3/envs/swedev_matze__pkgconfig-70/lib/python3.11/importlib/__init__.py:126: in import_module\n    return _bootstrap._gcd_import(name[level:], package, level)\ntest_0.py:3: in <module>\n    from pkgconfig.pkgconfig import Config\nE   ImportError: cannot import name 'Config' from 'pkgconfig.pkgconfig' (/mnt/nvme/playground/matze__pkgconfig-70_matze_pkgconfig_480e92f4c34dd0bc1ef243f23cdd98c1f2905ac6/pkgconfig/pkgconfig/pkgconfig.py)\n=========================== short test summary info ============================\nERROR test_0.py\n!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!\n=============================== 1 error in 0.09s ===============================\n", "stderr": "", "exit_code": 0}}
```
## Result
```
<class>pkgconfig.pkgconfig.Config</class>
```

## Error Message
```
"output": {{"stdout": "============================= test session starts ==============================\ncollecting ... collected 0 items / 1 error\n\n==================================== ERRORS ====================================\n__________________________ ERROR collecting example_module.py __________________________\nTypeError: example_function() missing 2 required positional arguments: 'param1' and 'param2'\n=========================== short test summary info ============================\nERROR example_module.py\n!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!\n=============================== 1 error in 0.04s ===============================\n", "stderr": "", "exit_code": 0}}
```

## Result
```
<function>example_module.example_function(param1, param2)</function>
```

(Please note the api should **directly** raise the error!)
(Remind again! For safety, you should never use `os.system` in the testcase code. If you want to operate on system, you should use the commands in setup commands!)
# Task
## Error Message
```
{}
```

## Result
"""