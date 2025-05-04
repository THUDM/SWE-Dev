from abc import ABC, abstractmethod

from swedev.utils.postprocess import extract_code_blocks, extract_locs_for_files
from swedev.utils.preprocess import get_full_file_paths_and_classes_and_functions, line_wrap_content, show_project_structure
from swedev.utils.utils import call
import os
from swedev.config import Config

class FileLocalizer(ABC):
    """Abstract class for file localizer"""
    def __init__(self, instance_id, structure, problem_statement, **kwargs):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement

    @abstractmethod
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        pass

class LLMFileLocalizer(FileLocalizer):
    """File localizer using LLM"""
    obtain_relevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of files that one would need to edit to fix the problem.
**You should wrap your response in triple backticks.**

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

Please only provide the full path and return at most 5 files.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
For example:
```
file1.py
file2.py
```
"""

    obtain_relevant_code_prompt = """
Please look through the following GitHub problem description and file and provide a set of locations that one would need to edit to fix the problem.
**You should wrap your response in triple backticks.**

### GitHub Problem Description ###
{problem_statement}

###

### File: {file_name} ###
{file_content}

###

Please provide either the class, the function name or line numbers that need to be edited. (Don't forget to wrap your response in triple backticks)
### Example 1:
```
class: MyClass
```
### Example 2:
```
function: my_function
```
### Example 3:
```
line: 10
line: 24
```

Return just the location(s)
"""
    file_content_template = """
### File: {file_name} ###
{file_content}
"""
    file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
"""
    obtain_relevant_code_combine_top_n_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class names, function or method names, or exact line numbers that require modification.
**You should wrap your response in triple backticks.**

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class name, function or method name, or the exact line numbers that need to be edited. **You should wrap your response in triple backticks.**
### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

Return just the location(s) (Don't forget to wrap your response in triple backticks)
"""
    obtain_relevant_code_graph_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
You will also be given a list of function/class dependencies to help you understand how functions/classes in relevant files fit into the rest of the codebase.
The locations can be specified as class names, function or method names, or exact line numbers that require modification. 
**You should wrap your response in triple backticks.**

### GitHub Problem Description ###
{problem_statement}

### Related Files ###
{file_contents}

### Function/Class Dependencies ###
{code_graph}

###

Please provide the class name, function or method name, or the exact line numbers that need to be edited.
### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

Return just the location(s) (Don't forget to wrap your response in triple backticks)
"""

    obtain_relevant_code_combine_top_n_no_line_number_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class, method, or function names that require modification. 
**You should wrap your response in triple backticks.**

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class, method, or function names that need to be edited.
### Examples:
```
full_path1/file1.py
function: my_function1
class: MyClass1

full_path2/file2.py
function: MyClass2.my_method
class: MyClass3

full_path3/file3.py
function: my_function2
```

Return just the location(s) (Don't forget to wrap your response in triple backticks)
"""
    obtain_relevant_functions_from_compressed_files_prompt = """
Please look through the following GitHub problem description and the skeleton of relevant files.
Provide a thorough set of locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related functions and classes.
**You should wrap your response in triple backticks.**

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide locations as either the class or the function name.
### Examples:
```
full_path1/file1.py
class: MyClass1

full_path2/file2.py
function: MyClass2.my_method

full_path3/file3.py
function: my_function
```

Return just the location(s) (Don't forget to wrap your response in triple backticks)
"""
    obtain_relevant_functions_and_vars_from_compressed_files_prompt_more = """
Please look through the following GitHub Problem Description and the Skeleton of Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Skeleton of Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
**You should wrap your response in triple backticks.**
### Examples:
```
full_path1/file1.py
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.py
variable: my_var
function: MyClass3.my_method

full_path3/file3.py
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

Return just the locations. (Don't forget to wrap your response in triple backticks)
"""

    def __init__(self, instance_id, structure, problem_statement, **kwargs):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 300

    def _parse_model_return_lines(self, content: str) -> list[str]:
        return content.strip().split("\n")

    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        """Localize the problem in the codebase"""
        found_files = []
        message = self.obtain_relevant_files_prompt.format(
            problem_statement=self.problem_statement,
            structure=show_project_structure(self.structure).strip(),
        ).strip()
        if mock:
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": 0, # dummy tokens
                },
            }
            return [], {"raw_output_loc": ""}, traj
        
        # Get model and URL from Config class
        localizer_model = Config.Localizer.model
        localizer_base_url = Config.Localizer.base_url
        
        resp, _ = call(
            model=localizer_model,
            messages=[{"role": "user", "content": message}],
            temperature=0.0,
            max_tokens=self.max_tokens,
            top_p=1.0,
            base_url=localizer_base_url,
        )
        traj = {
            "prompt": message,
            "response": resp,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }
        model_found_files = self._parse_model_return_lines(resp)
        
        files, classes, functions = get_full_file_paths_and_classes_and_functions(self.structure)
        for file_content in files:
            file = file_content[0]
            if file in model_found_files:
                found_files.append(file)

        # sort based on order of appearance in model_found_files
        found_files = sorted(found_files, key=lambda x: model_found_files.index(x))

        # print("Found files:", found_files)
        return (
            found_files,
            {"raw_output_files": resp},
            traj,
        )

    def localize_function_for_files(
        self, file_names, mock=False
    ) -> tuple[list, dict, dict]:
        """Localize the function in the codebase"""
        files, classes, functions = get_full_file_paths_and_classes_and_functions(self.structure)
        max_num_files = len(file_names)
        while True:
            # added small fix to prevent too many tokens
            contents = []
            for file_name in file_names[:max_num_files]:
                for file_content in files:
                    if file_content[0] == file_name:
                        content = "\n".join(file_content[1])
                        file_content = line_wrap_content(content)
                        contents.append(
                            self.file_content_template.format(
                                file_name=file_name, file_content=file_content
                            )
                        )
                        break
                else:
                    raise ValueError(f"File {file_name} does not exist.")

            file_contents = "".join(contents)
            max_num_files -= 1

        message = self.obtain_relevant_code_combine_top_n_prompt.format(
            problem_statement=self.problem_statement,
            file_contents=file_contents,
        ).strip()
        if mock:
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": calc_tokens(message, model),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        # Get model and URL from Config class
        localizer_model = Config.Localizer.model
        localizer_base_url = Config.Localizer.base_url
        
        resp, _ = call(
            model=localizer_model, 
            messages=[{"role": "user", "content": message}],
            temperature=0.0,
            max_tokens=self.max_tokens,
            top_p=1.0,
            base_url=localizer_base_url,
        )
        traj = {
            "prompt": message,
            "response": resp,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }

        model_found_locs = extract_code_blocks(resp)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names
        )
        return model_found_locs_separated, {"raw_output_loc": resp}, traj