import ast
import difflib
import os
import re


def api_formatter(apis):
    """
    Format the API signature to a more readable format.
    """
    ret = ''
    for api in apis:
        ret += f"{api['file']}: {api['signature']}\n"
    return ret

def calculate_similarity(func_signature_1, func_signature_2, type=None):
    """
    Calculate the similarity between two function signatures.
    The similarity is based on both the function name and the parameter list.
    """
    try:
        def parse_signature(signature):
                if type == "function":
                    name, args = signature.split("(", 1)
                    args = '(' + args
                    args = args.split(",") if args.strip() else []
                    args = [arg.strip() for arg in args]
                    return name.strip(), args
                elif type == "class":
                    return signature, []
                else:
                    raise NotImplementedError

        name1, args1 = parse_signature(func_signature_1)
        name2, args2 = parse_signature(func_signature_2)

        name_similarity = difflib.SequenceMatcher(None, name1, name2).ratio()
        arg_count_similarity = 1 - abs(len(args1) - len(args2)) / max(len(args1), len(args2), 1)
        common_args = len(set(args1) & set(args2))
        total_args = len(set(args1) | set(args2))
        arg_name_similarity = common_args / total_args if total_args > 0 else 0

        total_similarity = 0.6 * name_similarity + 0.2 * arg_count_similarity + 0.2 * arg_name_similarity
        return total_similarity
    except:
        return 0

def find_top_similar_apis(target_api, api_list, type=None, top_n=20):
    """
    Find the top N most similar APIs to the target API.
    
    :param target_api: The target API signature, e.g., "get_random_color():"
    :param api_list: A list of dictionaries, each containing 'file' and 'signature'
    :param top_n: The number of most similar APIs to return (default: 20)
    :return: A list of dictionaries with 'file', 'signature', and 'similarity'
    """
    similarities = []

    for api in api_list:
        file = api['file']
        signature = api['signature']
        similarity = calculate_similarity(target_api, signature, type)
        similarities.append({'file': file, 'signature': signature, 'similarity': similarity})

    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_n]

def extract_classes_and_functions(file_path, root_path=None):
    """
    Extract classes and their methods, as well as standalone functions, with full paths, from a Python file.

    :param file_path: Path to the Python file to parse.
    :param root_path: Optional root directory to calculate the module path (for packages).
    :return: A list of dictionaries with 'type', 'name', and 'args' for classes and functions.
    """
    with open(file_path, 'r', encoding='latin-1') as file:
        content = file.read()

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"SyntaxError in file {file_path}: {e}")
        return []  
    except Exception as e:
        print(f"Error in file {file_path}: {e}")
        return []  

    if root_path:
        relative_path = os.path.relpath(file_path, root_path)
        module_name = os.path.splitext(relative_path.replace(os.sep, '.'))[0]
    else:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
    signs = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef): 
            class_name = f"{module_name}.{node.name}"
            # class_name = f"{node.name}"
            for func in node.body:
                if isinstance(func, ast.FunctionDef):
                    args = [arg.arg for arg in func.args.args]
                    signs.append({
                        "type": "method",
                        "name": f"{func.name}",
                        "name": f"{class_name}.{func.name}",
                        "args": args,
                    })
                    
        elif isinstance(node, ast.FunctionDef): 
            args = [arg.arg for arg in node.args.args]
            signs.append({
                "type": "function",
                "name": f"{module_name}.{node.name}",
                "args": args,
            })
    return signs

def save_classes_and_functions_to_file(output_file, classes_and_functions_by_file):
    """Save extracted classes and functions to a file."""
    with open(output_file, 'w', encoding='utf-8') as file:
        for filename, data in classes_and_functions_by_file.items():
            file.write(f"<{filename}>\n")
            if data['standalone_functions']:
                file.write("Standalone Functions:\n")
                file.write("\n".join(data['standalone_functions']) + "\n\n")

            for class_name, methods in data['classes'].items():
                file.write(f"Class {class_name}:\n")
                file.write("\n".join(f"  {method}" for method in methods) + "\n\n")

def extract_classes_and_functions_from_directory(root_dir):
    """Extract classes and functions from all Python files in a given directory."""
    all_results = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(file_path, root_dir)
                results = extract_classes_and_functions(file_path)
                if results:
                    all_results.append({
                        "file": relative_path,
                        "content": results
                    })

    return all_results

def generate_signatures(extracted_data, type=None):
    """
    Generate complete function/method signatures from the extracted data,
    filtered by the specified type.
    
    :param extracted_data: A list of dictionaries containing file data, content, and extracted items.
    :param type: The type of signature to generate ("function" or "class").
    :return: A list of dictionaries with file name, type, and signature.
    """
    used, signatures = [], []

    for file_data in extracted_data:
        file_name = file_data["file"]
        content = file_data["content"]

        for item in content:
            if type == "function" and (item["type"] == "function" or item["type"] == "method"):
                func_name = item["name"]
                args = ", ".join(item["args"])
                signature = f"{func_name}({args})"
                signatures.append({
                    "file": file_name,
                    "type": "function",
                    "signature": signature
                })
            elif type == "class" and item["type"] == "method":                
                class_name = ".".join(item["name"].split(".")[:-1])
                signature = f"{class_name}"
                if signature in used:
                    continue
                used.append(signature)                
                signatures.append({
                    "file": file_name,
                    "type": "class",
                    "signature": signature
                })
    
    return signatures

def parse_api(response):
    """
    Parse the API response and extract the classes and functions.
    The response is wrapped by <function></function>, <class></class> and <empty></empty> for functions, classes and empty lines respectively.
    :param response: The API response from the server
    :return: api like  "get_random_color()"
    """ 
    function_pattern = r"<function>(.*?)</function>"
    class_pattern = r"<class>(.*?)</class>"
    function_match = re.search(function_pattern, response)
    class_match = re.search(class_pattern, response)
    if function_match:
        return "function", function_match.group(1)
    elif class_match:
        return "class", class_match.group(1)
    return "empty", None
    
    
if __name__ == "__main__":
    root_directory = "."
    output_file = "classes_and_functions.txt"
    api_dict = generate_signatures(extract_classes_and_functions_from_directory(root_directory), type="function")
    print(api_dict[:10])
    print(f"Saved classes and functions to {output_file}")