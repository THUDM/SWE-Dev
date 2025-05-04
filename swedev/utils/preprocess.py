import ast

def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            # print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            # print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )
    return class_info, function_names, file_content.splitlines()

def line_wrap_content(
    content: str,
    context_intervals=None,
    add_space=False,
    no_line_number=False,
    sticky_scroll=False,
):
    """add n| to each line, where n increases"""

    def is_scope(line):
        return line.startswith("class ") or line.strip().startswith("def ")

    lines = content.split("\n")
    new_lines = []
    if context_intervals is None or context_intervals == []:
        context_intervals = [(0, len(lines))]

    prev_scopes = []
    line_format = "{line}"
    if not no_line_number:
        line_format = (
            "{line_number}|{line}" if not add_space else "{line_number}| {line} "
        )
    for interval in context_intervals:
        min_line, max_line = interval

        if min_line != 0:
            new_lines.append("...")

        scopes = []
        for i, line in enumerate(lines):
            if sticky_scroll:
                # add current line to scope if necessary
                if is_scope(line):
                    indent_level = len(line) - len(line.lstrip())
                    while scopes and scopes[-1]["indent_level"] >= indent_level:
                        scopes.pop()
                    scopes.append(
                        {"line": line, "line_number": i, "indent_level": indent_level}
                    )

            if min_line != -1 and i < min_line - 1:
                continue
            if sticky_scroll and i == min_line - 1:
                # add scope lines
                last_scope_line = None
                for j, scope_line in enumerate(scopes):
                    # don't repeat previous scopes
                    if (
                        len(prev_scopes) > j
                        and prev_scopes[j]["line_number"] == scope_line["line_number"]
                    ):
                        continue
                    # don't repeat current line
                    if i == scope_line["line_number"]:
                        continue
                    new_lines.append(
                        line_format.format(
                            line_number=scope_line["line_number"] + 1,
                            line=scope_line["line"],
                        )
                    )
                    last_scope_line = scope_line["line_number"]
                if last_scope_line is not None and last_scope_line < i - 1:
                    new_lines.append("...")

            new_lines.append(line_format.format(line_number=i + 1, line=line))
            if max_line != -1 and i >= max_line - 1:
                break
        prev_scopes = scopes

    if max_line != len(lines):
        new_lines.append("...")

    return "\n".join(new_lines)

def show_project_structure(structure, spacing=0) -> str:
    """pprint the project structure"""

    pp_string = ""

    for key, value in structure.items():
        if "." in key and ".py" not in key:
            continue  # skip none python files
        if "." in key:
            pp_string += " " * spacing + str(key) + "\n"
        else:
            pp_string += " " * spacing + str(key) + "/" + "\n"
        if "classes" not in value:
            pp_string += show_project_structure(value, spacing + 4)

    return pp_string


def filter_out_test_files(structure):
    """filter out test files from the project structure"""
    for key, value in list(structure.items()):
        if key.startswith("test"):
            del structure[key]
        elif isinstance(value, dict):
            filter_out_test_files(value)


def filter_none_python(structure):
    for key, value in list(structure.items()):
        if (
            not "functions" in value.keys()
            and not "classes" in value.keys()
            and not "text" in value.keys()
        ) or not len(value.keys()) == 3:
            filter_none_python(value)

            if structure[key] == {}:
                del structure[key]
        else:
            if not key.endswith(".py"):
                del structure[key]


def get_full_file_paths_and_classes_and_functions(structure, current_path=""):
    """
    Recursively retrieve all file paths, classes, and functions within a directory structure.

    Arguments:
    structure -- a dictionary representing the directory structure
    current_path -- the path accumulated so far, used during recursion (default="")

    Returns:
    A tuple containing:
    - files: list of full file paths
    - classes: list of class details with file paths
    - functions: list of function details with file paths
    """
    files = []
    classes = []
    functions = []
    for name, content in structure.items():
        if isinstance(content, dict):
            if (
                not "functions" in content.keys()
                and not "classes" in content.keys()
                and not "text" in content.keys()
            ) or not len(content.keys()) == 3:
                # or guards against case where functions and classes are somehow part of the structure.
                next_path = f"{current_path}/{name}" if current_path else name
                (
                    sub_files,
                    sub_classes,
                    sub_functions,
                ) = get_full_file_paths_and_classes_and_functions(content, next_path)
                files.extend(sub_files)
                classes.extend(sub_classes)
                functions.extend(sub_functions)
            else:
                next_path = f"{current_path}/{name}" if current_path else name
                files.append((next_path, content["text"]))
                if "classes" in content:
                    for clazz in content["classes"]:
                        classes.append(
                            {
                                "file": next_path,
                                "name": clazz["name"],
                                "start_line": clazz["start_line"],
                                "end_line": clazz["end_line"],
                                "methods": [
                                    {
                                        "name": method["name"],
                                        "start_line": method["start_line"],
                                        "end_line": method["end_line"],
                                    }
                                    for method in clazz.get("methods", [])
                                ],
                            }
                        )
                if "functions" in content:
                    for function in content["functions"]:
                        function["file"] = next_path
                        functions.append(function)
        else:
            next_path = f"{current_path}/{name}" if current_path else name
            files.append(next_path)
    return files, classes, functions