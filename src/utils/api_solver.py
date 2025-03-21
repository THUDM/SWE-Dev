import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class APISignature:
    module: str
    name: str
    params: List[Tuple[str, str]]  # [(param_name, param_type), ...]
    return_type: str
    full_signature: str
    doc_string: Optional[str] = None

@dataclass
class CodeContext:
    imports: Dict[str, Set[str]]  # {module_name: {imported_names}}
    current_module: str
    current_class: Optional[str]
    current_function: Optional[str]
    local_variables: Dict[str, str]  # {var_name: type_hint}
    used_apis: List[str]
    file_path: Path
    code_block: str

@dataclass
class APIMatch:
    api: APISignature
    score: float
    match_reason: str

class APIContextAnalyzer:
    def __init__(self, code_content: str, file_path: Path):
        self.code_content = code_content
        self.file_path = file_path
        self.tree = ast.parse(code_content)
        
    def extract_context(self, line_no: int) -> CodeContext:
        context = self._initialize_context()
        self._analyze_imports(self.tree, context)
        
        current_node = self._find_node_at_line(self.tree, line_no)
        if current_node:
            self._analyze_node_context(current_node, context)
            
        return context
    
    def _initialize_context(self) -> CodeContext:
        return CodeContext(
            imports={},
            current_module=self.file_path.stem,
            current_class=None,
            current_function=None,
            local_variables={},
            used_apis=[],
            file_path=self.file_path,
            code_block=""
        )
    
    def _analyze_imports(self, tree: ast.AST, context: CodeContext):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    context.imports[name.name] = {name.name}
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = {n.name for n in node.names}
                context.imports[module] = names
    
    def _find_node_at_line(self, tree: ast.AST, line_no: int) -> Optional[ast.AST]:
        result = None
        
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                if node.lineno <= line_no <= getattr(node, 'end_lineno', node.lineno):
                    if result is None or ast.get_source_segment(self.code_content, node) < ast.get_source_segment(self.code_content, result):
                        result = node
                        
        return result
    
    def _analyze_node_context(self, node: ast.AST, context: CodeContext):
        for parent in ast.walk(self._get_parent(node)):
            if isinstance(parent, ast.ClassDef):
                context.current_class = parent.name
            elif isinstance(parent, ast.FunctionDef):
                context.current_function = parent.name
                
        scope = self._get_scope(node)
        for var_node in ast.walk(scope):
            if isinstance(var_node, ast.AnnAssign):
                if isinstance(var_node.target, ast.Name):
                    context.local_variables[var_node.target.id] = ast.unparse(var_node.annotation)
        self._extract_used_apis(scope, context)
        context.code_block = ast.get_source_segment(self.code_content, scope)
    
    def _get_parent(self, node: ast.AST) -> Optional[ast.AST]:
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                if child == node:
                    return parent
        return None
    
    def _get_scope(self, node: ast.AST) -> ast.AST:
        current = node
        while current:
            if isinstance(current, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                return current
            current = self._get_parent(current)
        return self.tree
    
    def _extract_used_apis(self, scope: ast.AST, context: CodeContext):
        for node in ast.walk(scope):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    api_call = ast.unparse(node.func)
                    context.used_apis.append(api_call)

class SemanticMatcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.api_embeddings = {}
        
    def prepare_embeddings(self, apis: List[APISignature]):
        for api in apis:
            embedding = self._get_api_embedding(api)
            self.api_embeddings[api.full_signature] = embedding
    
    def _get_api_embedding(self, api: APISignature) -> np.ndarray:
        text = self._api_to_text(api)
        return self.model.encode(text)
    
    def _api_to_text(self, api: APISignature) -> str:
        parts = []
        
        parts.append(f"{api.module} {api.name}")
        
        param_desc = [f"{name}: {type_}" for name, type_ in api.params]
        if param_desc:
            parts.append("parameters: " + ", ".join(param_desc))
            
        parts.append(f"returns {api.return_type}")
        
        if api.doc_string:
            parts.append(api.doc_string)
            
        return " ".join(parts)
    
    def find_matches(self, query: str, top_k: int = 5) -> List[Tuple[APISignature, float]]:
        query_embedding = self.model.encode(query)
        
        similarities = []
        for api_sig, api_embedding in self.api_embeddings.items():
            sim = self._cosine_similarity(query_embedding, api_embedding)
            similarities.append((api_sig, sim))
            
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class StructureMatcher:
    def __init__(self, apis: List[APISignature]):
        self.apis = apis
        self.param_type_index = self._build_param_type_index()
        
    def _build_param_type_index(self) -> Dict[str, List[APISignature]]:
        index = defaultdict(list)
        for api in self.apis:
            for _, param_type in api.params:
                index[param_type].append(api)
        return index
    
    def find_matches(self, context: CodeContext) -> List[Tuple[APISignature, float]]:
        matches = []
        
        for api in self.apis:
            score = self._calculate_structure_score(api, context)
            if score > 0:
                matches.append((api, score))
                
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def _calculate_structure_score(self, api: APISignature, context: CodeContext) -> float:
        score = 0.0
        
        available_types = set(context.local_variables.values())
        required_types = {param_type for _, param_type in api.params}
        if required_types:
            matching_types = required_types.intersection(available_types)
            score += len(matching_types) / len(required_types)
        
        if context.current_class:
            if api.name.startswith(f"_{context.current_class}"):
                score += 0.3
                
        return score

class APIMatchEngine:
    def __init__(self, apis: List[APISignature]):
        self.apis = apis
        self.semantic_matcher = SemanticMatcher()
        self.semantic_matcher.prepare_embeddings(apis)
        self.structure_matcher = StructureMatcher(apis)
        
    def find_similar_apis(self, 
                         error_msg: str, 
                         code_content: str, 
                         file_path: Path,
                         line_no: int) -> List[APIMatch]:
        context_analyzer = APIContextAnalyzer(code_content, file_path)
        context = context_analyzer.extract_context(line_no)
        
        query = self._extract_query_from_error(error_msg)
        
        semantic_matches = self.semantic_matcher.find_matches(query)
        structure_matches = self.structure_matcher.find_matches(context)
        
        final_matches = self._combine_matches(
            semantic_matches,
            structure_matches,
            context
        )
        
        return self._format_matches(final_matches)
    
    def _extract_query_from_error(self, error_msg: str) -> str:
        attribute_match = re.search(r"AttributeError: [^']*'([^']*)'", error_msg)
        if attribute_match:
            return attribute_match.group(1)
            
        name_match = re.search(r"NameError: name '([^']*)' is not defined", error_msg)
        if name_match:
            return name_match.group(1)
            
        return error_msg
    
    def _combine_matches(self,
                        semantic_matches: List[Tuple[str, float]],
                        structure_matches: List[Tuple[APISignature, float]],
                        context: CodeContext) -> List[Tuple[APISignature, float, str]]:
        combined_scores = defaultdict(lambda: {'score': 0.0, 'reasons': []})
        
        for api_sig, score in semantic_matches:
            combined_scores[api_sig]['score'] += score * 0.6
            if score > 0.5:
                combined_scores[api_sig]['reasons'].append('semantic_similarity')
                
        for api, score in structure_matches:
            combined_scores[api.full_signature]['score'] += score * 0.4
            if score > 0.3:
                combined_scores[api.full_signature]['reasons'].append('structure_match')
                
        self._apply_context_rules(combined_scores, context)
        
        results = []
        for api_sig, details in combined_scores.items():
            if details['score'] > 0.2:
                api = next(api for api in self.apis if api.full_signature == api_sig)
                results.append((api, details['score'], ', '.join(details['reasons'])))
                
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _apply_context_rules(self, scores: Dict[str, Dict], context: CodeContext):
        for api_sig in scores:
            api = next(api for api in self.apis if api.full_signature == api_sig)
            
            if api.module == context.current_module:
                scores[api_sig]['score'] *= 1.2
                scores[api_sig]['reasons'].append('same_module')
                
            if api.module in context.imports:
                scores[api_sig]['score'] *= 1.1
                scores[api_sig]['reasons'].append('imported')
                
            if any(used_api.endswith(api.name) for used_api in context.used_apis):
                scores[api_sig]['score'] *= 1.15
                scores[api_sig]['reasons'].append('usage_pattern')
    
    def _format_matches(self, matches: List[Tuple[APISignature, float, str]]) -> List[APIMatch]:
        return [APIMatch(api=api, score=score, match_reason=reason)
                for api, score, reason in matches]

def find_similar_apis(error_msg: str, code_content: str, file_path: str, line_no: int, apis: List[APISignature]) -> List[APIMatch]:
    engine = APIMatchEngine(apis)
    matches = engine.find_similar_apis(
        error_msg=error_msg,
        code_content=code_content,
        file_path=Path(file_path),
        line_no=line_no
    )
    return matches

if __name__ == "__main__":
    sample_apis = [
        APISignature(
            module="data.processing",
            name="process_data",
            params=[("data", "DataFrame"), ("columns", "List[str]")],
            return_type="DataFrame",
            full_signature="data.processing.process_data(data: DataFrame, columns: List[str]) -> DataFrame",
            doc_string="Process data with specified columns"
        ),
    ]
    
    error_msg = "AttributeError: module 'data.processing' has no attribute 'process'"
    
    code_content = """
    import pandas as pd
    from data.processing import process_data
    
    df = pd.DataFrame()
    result = process(df, ['col1', 'col2'])
    """
    
    matches = find_similar_apis(
        error_msg=error_msg,
        code_content=code_content,
        file_path="test.py",
        line_no=5,
        apis=sample_apis
    )
    
    for match in matches:
        print(f"API: {match.api.full_signature}")
        print(f"Score: {match.score:.2f}")
        print(f"Reason: {match.match_reason}")
        print()