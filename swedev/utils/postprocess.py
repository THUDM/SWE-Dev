import ast
import re


def extract_code_blocks(text):
    pattern = r"```\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        if "```" in text:
            # handle the case where the code block is not complete
            return [text.split("```", 1)[-1].strip()]
    return matches


def extract_locs_for_files(locs, file_names):
    results = {fn: [] for fn in file_names}
    current_file_name = None
    for loc in locs:
        for line in loc.splitlines():
            if line.strip().endswith(".py"):
                current_file_name = line.strip()
            elif line.strip() and any(
                line.startswith(w)
                for w in ["line:", "function:", "class:", "variable:"]
            ):
                if current_file_name in results:
                    results[current_file_name].append(line)
                else:
                    pass
    return [["\n".join(results[fn])] for fn in file_names]

def get_codebody(source_code: str, omit: bool = False) -> str:
    class CodeModifier(ast.NodeTransformer):
        def visit_ClassDef(self, node):
            if node.name.startswith('Test') or node.name.endswith('Test'):
                return None
            
            docstring = ast.get_docstring(node)
            if docstring and omit:
                new_body = [ast.Expr(value=ast.Constant(value=docstring))]
                new_body.append(ast.Expr(value=ast.Constant(value="<omit function content for briefness>")))
                new_node = ast.ClassDef(
                    name=node.name,
                    bases=node.bases,
                    keywords=node.keywords,
                    body=new_body,
                    decorator_list=node.decorator_list
                )
                return new_node
            return node
            
        def visit_FunctionDef(self, node):
            if node.name.startswith('test_') or 'test' in node.name.lower():
                return None
            
            docstring = ast.get_docstring(node)
            if docstring and omit:
                new_body = [ast.Expr(value=ast.Constant(value=docstring))]
                new_body.append(ast.Expr(value=ast.Constant(value="<omit function content for briefness>")))
                new_node = ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=new_body,
                    decorator_list=node.decorator_list,
                    returns=node.returns
                )
                return new_node
            return node
            
        def visit_Import(self, node):
            test_imports = ['pytest', 'unittest', 'mock']
            new_names = [n for n in node.names if n.name not in test_imports]
            if not new_names:
                return None
            node.names = new_names
            return node
        
        def visit_ImportFrom(self, node):
            try:
                if any(x in node.module.lower() for x in ['test', 'mock']):
                    return None
                return node
            except Exception as e:
                return None

    tree = ast.parse(source_code)
    transformer = CodeModifier()
    cleaned_tree = transformer.visit(tree)
    ast.fix_missing_locations(cleaned_tree)
    return ast.unparse(cleaned_tree)

if __name__ == "__main__":
    example_code = '''
    def calculate_sum(a, b):
        """
        Args:
            a: First Number
            b: Second Number
            
        Returns:
            Sum of the numbers
        """
        return a + b

    def no_doc_function(x):
        return x * 2

    def test_calculate():
        assert calculate_sum(1, 2) == 3
        
    class Calculator:
        def add(self, a, b):
            return a + b
            
    class NoDocClass:
        def method(self):
            pass
            
    class TestCalculator:
        def test_add(self):
            calc = Calculator()
            assert calc.add(1, 2) == 3
    '''

    print("With omit=True:")
    print(get_codebody(example_code, omit=True))

    print("\nWith omit=False:")
    print(get_codebody(example_code, omit=False))