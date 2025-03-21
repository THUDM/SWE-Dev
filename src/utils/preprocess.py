import ast
import json
import os

import libcst as cst
from src.utils.parse_global_var import GlobalVariableVisitor


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


def merge_intervals(intervals):
    # intervals inclusive
    if not intervals:
        return []

    # Sort the intervals based on the starting value of each tuple
    intervals.sort(key=lambda interval: interval[0])

    merged_intervals = [intervals[0]]

    for current in intervals[1:]:
        last = merged_intervals[-1]

        # Check if there is overlap
        if current[0] <= last[1]:
            # If there is overlap, merge the intervals
            merged_intervals[-1] = (last[0], max(last[1], current[1]))
        else:
            # If there is no overlap, just add the current interval to the result list
            merged_intervals.append(current)

    return merged_intervals


def parse_global_var_from_code(file_content: str) -> dict[str, dict]:
    """Parse global variables."""
    try:
        tree = cst.parse_module(file_content)
    except:
        return file_content

    wrapper = cst.metadata.MetadataWrapper(tree)
    visitor = GlobalVariableVisitor()
    wrapper.visit(visitor)

    global_assigns = {}
    for assign_stmt, start_pos, end_pos in visitor.global_assigns:
        for t in assign_stmt.body:
            try:
                targets = [t.targets[0].target.value]
            except:
                try:
                    targets = t.targets[0].target.elements
                    targets = [x.value.value for x in targets]
                except:
                    targets = []
            for target_var in targets:
                global_assigns[target_var] = {
                    "start_line": start_pos.line,
                    "end_line": end_pos.line,
                }
    return global_assigns

def transfer_arb_locs_to_locs(
    locs,
    structure,
    pred_file,
    context_window=10,
    loc_interval=False,
    fine_grain_only=False,
    remove_line=False,
    file_content="",
    verbose=False,
) -> tuple[list, list]:
    if structure is None:
        class_info, function_names, file_lines = parse_python_file("", file_content)
        structure = {}
        structure[pred_file] = {
            "classes": class_info,
            "functions": function_names,
            "text": file_lines,
        }

    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    line_loc = []
    if isinstance(locs, str):
        # if its a single loc
        locs = [locs]
    global_vars = parse_global_var_from_code(file_content)
    unrecognized_locs = []
    for model_pred_locs in locs:
        current_class_name = ""
        for loc in model_pred_locs.splitlines():
            # handle cases like "class: MyClass.my_method"
            if loc.startswith("class: ") and "." not in loc:
                loc = loc[len("class: ") :].strip()
                relevant_class = [
                    clazz
                    for clazz in classes
                    if clazz["file"] == pred_file and clazz["name"] == loc
                ]

                if len(relevant_class) == 0:
                    unrecognized_locs.append(loc)
                else:
                    line_loc.append(
                        (relevant_class[0]["start_line"], relevant_class[0]["end_line"])
                    )
                    current_class_name = loc

            elif loc.startswith("function: ") or "." in loc:
                full_loc = loc
                loc = loc.split(":", 1)[-1].strip()

                if "." in loc:
                    # assume its a method within a class
                    method_name = loc.split(".")[1]
                    class_name = loc.split(".")[0]

                    relevant_class = [
                        clazz
                        for clazz in classes
                        if clazz["file"] == pred_file and clazz["name"] == class_name
                    ]
                    if len(relevant_class) == 0:
                        unrecognized_locs.append(loc)
                    else:
                        relevant_method = [
                            method
                            for method in relevant_class[0]["methods"]
                            if method["name"] == method_name
                        ]
                        if len(relevant_method) == 0:
                            unrecognized_locs.append(loc)
                        else:
                            line_loc.append(
                                (
                                    relevant_method[0]["start_line"],
                                    relevant_method[0]["end_line"],
                                )
                            )

                else:
                    relevant_function = [
                        function
                        for function in functions
                        if function["file"] == pred_file and function["name"] == loc
                    ]
                    if len(relevant_function) == 0:
                        if current_class_name != "":
                            # check if its a method
                            relevant_class = [
                                clazz
                                for clazz in classes
                                if clazz["file"] == pred_file
                                and clazz["name"] == current_class_name
                            ]
                            relevant_method = [
                                method
                                for method in relevant_class[0]["methods"]
                                if method["name"] == loc
                            ]
                            if len(relevant_method) == 0:
                                unrecognized_locs.append(loc)
                            else:
                                line_loc.append(
                                    (
                                        relevant_method[0]["start_line"],
                                        relevant_method[0]["end_line"],
                                    )
                                )
                        else:
                            # look for it in any class
                            relevant_method = []
                            for clazz in classes:
                                if clazz["file"] == pred_file:
                                    relevant_method.extend(
                                        [
                                            method
                                            for method in clazz["methods"]
                                            if method["name"] == loc
                                        ]
                                    )

                            if len(relevant_method) == 1:
                                line_loc.append(
                                    (
                                        relevant_method[0]["start_line"],
                                        relevant_method[0]["end_line"],
                                    )
                                )
                    else:
                        line_loc.append(
                            (
                                relevant_function[0]["start_line"],
                                relevant_function[0]["end_line"],
                            )
                        )
            elif loc.startswith("line: "):
                if remove_line:
                    continue
                loc = loc[len("line: ") :].strip().split()[0]
                try:
                    # line_loc.append(int(loc))
                    line_loc.append((int(loc), int(loc)))
                except:
                    continue
            elif loc.startswith("variable:"):
                vars = loc[len("variable:") :].strip().split()
                for v in vars:
                    if v in global_vars:
                        line_loc.append(
                            (global_vars[v]["start_line"], global_vars[v]["end_line"])
                        )
            else:
                if loc.strip():
                    unrecognized_locs.append(loc)
                # assert False

    # Fine-grained-only loc: Remove intervals that are supersets of another.
    if fine_grain_only:
        filtered_line_loc = []
        for st, en in line_loc:
            if filtered_line_loc:
                last_st, last_en = filtered_line_loc[-1]
                # If the current interval is a more fine-grained loc, remove the superset.
                if last_st <= st and en <= last_en:
                    filtered_line_loc.pop()
            filtered_line_loc.append((st, en))
        line_loc = filtered_line_loc

    # compute max min
    for file_content in files:
        if file_content[0] == pred_file:
            content = file_content[1]
            break

    if len(line_loc) == 0:
        return [], []

    # max_line = min(max(line_loc) + context_window, len(content))
    # min_line = max(min(line_loc) - context_window, 0)
    #
    # return line_loc, max_line, min_line

    if verbose:
        print("Unrecognized locs:")
        for loc in unrecognized_locs:
            print(loc)

    # compute overlapping locations instead
    if loc_interval:
        contextual_line_loc = []
        for loc in line_loc:
            # Clip the context window to the beginning and end of the file
            max_line = max(min(loc[1] + context_window, len(content)), 0)
            min_line = min(max(loc[0] - context_window, 0), len(content))
            contextual_line_loc.append((min_line, max_line))

        return line_loc, merge_intervals(contextual_line_loc)
    else:
        # defaulting to max min
        max_line = min(max([loc[1] for loc in line_loc]) + context_window, len(content))
        min_line = max(min([loc[0] for loc in line_loc]) - context_window, 0)

        return line_loc, [(min_line, max_line)]

def compile_gt_locations(gt_location: dict) -> tuple[list, set, set, set]:
    """mostly serves a way to check what are the gt locations in gt patch"""
    edits = gt_location["edits"]

    lines, classes, methods, functions = [], set(), set(), set()

    adds = set()

    for edit in edits:
        for clazz in edit["class_names"]:
            classes.add(clazz)

        for method in edit["method_names"]:
            methods.add(method)

        for function in edit["function_names"]:
            functions.add(function)

        if edit["type"] == "add":
            adds.add(edit["line"])
        else:
            lines.append(edit["line"])

    # handle the added lines
    add_intervals = [(i, i + 1) for i in adds]
    add_intervals = merge_intervals(add_intervals)
    for st, en in add_intervals:
        lines.append(st)
    lines = list(set(lines))

    # sort the lines
    lines = sorted(lines)

    return lines, classes, methods, functions


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


def filter_proposed_files(proposed_files, repo_structure):
    """
    Filter proposed files against a given repository structure.

    Arguments:
    proposed_files -- list of proposed files with instance IDs
    repo_structure -- list of repository structures with instance IDs

    Returns:
    A list of dictionaries with instance IDs and valid files matching the repository structure.
    """
    instance_to_files = {
        entry["instance_id"]: entry["files"] for entry in proposed_files
    }
    instance_to_structure = {
        entry["instance_id"]: entry["structure"] for entry in repo_structure
    }
    filtered_files = []
    for instance_id, files in instance_to_files.items():
        if instance_id in instance_to_structure:
            repo_files, _, _ = get_full_file_paths_and_classes_and_functions(
                instance_to_structure[instance_id]
            )
            repo_files_set = set(repo_files)
            valid_files = []
            for repo_file in repo_files_set:
                for proposed_file in files:
                    if proposed_file == repo_file.split("/")[-1]:
                        valid_files.append(repo_file)
            if valid_files:
                filtered_files.append(
                    {"instance_id": instance_id, "files": valid_files}
                )
    return filtered_files


def filter_proposed_classes(proposed_classes, repo_structure):
    """
    Filter proposed classes against a given repository structure.

    Arguments:
    proposed_classes -- list of proposed classes with instance IDs
    repo_structure -- list of repository structures with instance IDs

    Returns:
    A list of dictionaries with instance IDs and valid classes matching the repository structure.
    """
    instance_to_classes = {
        entry["instance_id"]: entry["classes"] for entry in proposed_classes
    }
    instance_to_structure = {
        entry["instance_id"]: entry["structure"] for entry in repo_structure
    }
    filtered_classes = []
    for instance_id, classes in instance_to_classes.items():
        if instance_id in instance_to_structure:
            _, repo_classes, _ = get_full_file_paths_and_classes_and_functions(
                instance_to_structure[instance_id]
            )
            repo_classes_set = {clazz["name"]: clazz["file"] for clazz in repo_classes}
            valid_classes = []
            for proposed_class in classes:
                if proposed_class in repo_classes_set:
                    valid_classes.append(
                        {
                            "name": proposed_class,
                            "file": repo_classes_set[proposed_class],
                        }
                    )
            if valid_classes:
                filtered_classes.append(
                    {"instance_id": instance_id, "classes": valid_classes}
                )
    return filtered_classes


def filter_proposed_methods(proposed_methods, repo_structure):
    """
    Filter proposed methods against a given repository structure.

    Arguments:
    proposed_methods -- list of proposed methods with instance IDs
    repo_structure -- list of repository structures with instance IDs

    Returns:
    A list of dictionaries with instance IDs and valid methods matching the repository structure.
    """
    instance_to_methods = {
        entry["instance_id"]: entry["methods"] for entry in proposed_methods
    }
    instance_to_structure = {
        entry["instance_id"]: entry["structure"] for entry in repo_structure
    }
    filtered_methods = []
    for instance_id, methods in instance_to_methods.items():
        if instance_id in instance_to_structure:
            _, repo_classes, _ = get_full_file_paths_and_classes_and_functions(
                instance_to_structure[instance_id]
            )
            valid_methods = []
            for repo_class in repo_classes:
                for method in methods:
                    if method in repo_class["methods"]:
                        valid_methods.append(
                            {
                                "class": repo_class["name"],
                                "method": method,
                                "file": repo_class["file"],
                            }
                        )
            if valid_methods:
                filtered_methods.append(
                    {"instance_id": instance_id, "methods": valid_methods}
                )
    return filtered_methods


def filter_proposed_functions(proposed_functions, repo_structure):
    """
    Filter proposed functions against a given repository structure.

    Arguments:
    proposed_functions -- list of proposed functions with instance IDs
    repo_structure -- list of repository structures with instance IDs

    Returns:
    A list of dictionaries with instance IDs and valid functions matching the repository structure.
    """
    instance_to_functions = {
        entry["instance_id"]: entry["functions"] for entry in proposed_functions
    }
    instance_to_structure = {
        entry["instance_id"]: entry["structure"] for entry in repo_structure
    }
    filtered_functions = []
    for instance_id, functions in instance_to_functions.items():
        if instance_id in instance_to_structure:
            _, _, repo_functions = get_full_file_paths_and_classes_and_functions(
                instance_to_structure[instance_id]
            )
            valid_functions = []
            for repo_function in repo_functions:
                for function in functions:
                    if isinstance(
                        repo_function["name"], dict
                    ):  # Why are there cases where this is not a dict?
                        if function == repo_function["name"].get("name", []):
                            valid_functions.append(
                                {"function": function, "file": repo_function["file"]}
                            )
            if valid_functions:
                filtered_functions.append(
                    {"instance_id": instance_id, "functions": valid_functions}
                )
    return filtered_functions


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


PROJECT_FILE_LOC = os.environ.get("PROJECT_FILE_LOC", None)

def get_repo_files(structure, filepaths: list[str]):
    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)
    file_contents = dict()
    for filepath in filepaths:
        content = None

        for file_content in files:
            if file_content[0] == filepath:
                content = "\n".join(file_content[1])
                file_contents[filepath] = content
                break

        assert content is not None, "file not found"
    return file_contents


def test_merge():
    # Example usage:
    input_tuples = [(1, 3), (2, 4), (5, 7), (6, 8)]
    merged_tuples = merge_intervals(input_tuples)
    assert merged_tuples == [(1, 4), (5, 8)]

    input_tuples = [(1, 5), (2, 3)]
    merged_tuples = merge_intervals(input_tuples)
    assert merged_tuples == [(1, 5)]

    input_tuples = [(1, 1)]
    merged_tuples = merge_intervals(input_tuples)
    assert merged_tuples == [(1, 1)]

    input_tuples = [(1, 1), (2, 3)]
    merged_tuples = merge_intervals(input_tuples)
    assert merged_tuples == [(1, 1), (2, 3)]


def test_interval_display():

    content = """
one
two
three
four
five
six
seven
eight
""".strip()

    x = line_wrap_content(content, [])
    print(x)

    print("============")

    x = line_wrap_content(content, [(1, 2), (4, 6), (7, 8)])
    print(x)

def construct_topn_file_context(
    file_to_locs,
    pred_files,
    file_contents,
    structure,
    context_window: int,
    loc_interval: bool = True,
    fine_grain_loc_only: bool = False,
    add_space: bool = False,
    sticky_scroll: bool = False,
    no_line_number: bool = True,
):
    """Concatenate provided locations to form a context.

    loc: {"file_name_1": ["loc_str_1"], ...}
    """
    file_loc_intervals = dict()
    topn_content = ""

    for pred_file, locs in file_to_locs.items():
        content = file_contents[pred_file]
        line_locs, context_intervals = transfer_arb_locs_to_locs(
            locs,
            structure,
            pred_file,
            context_window,
            loc_interval,
            fine_grain_loc_only,
            file_content=file_contents[pred_file] if pred_file in file_contents else "",
        )
        if len(line_locs) > 0:
            file_loc_content = line_wrap_content(
                content,
                context_intervals,
                add_space=add_space,
                no_line_number=no_line_number,
                sticky_scroll=sticky_scroll,
            )
            topn_content += f"### {pred_file}\n{file_loc_content}\n\n\n"
            file_loc_intervals[pred_file] = context_intervals

    return topn_content, file_loc_intervals

if __name__ == "__main__":
    test_merge()
    # test_interval_display()