import re
from dataclasses import dataclass
from difflib import get_close_matches
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.utils.utils import *


class ErrorType(Enum):
    ATTRIBUTE_ERROR = "AttributeError"
    NAME_ERROR = "NameError" 
    TYPE_ERROR = "TypeError"
    ASSERTION_ERROR = "AssertionError"
    IMPORT_ERROR = "ImportError"
    SYNTAX_ERROR = "SyntaxError"
    VALUE_ERROR = "ValueError"
    INDEX_ERROR = "IndexError"
    KEY_ERROR = "KeyError"
    RUNTIME_ERROR = "RuntimeError"
    OTHER = "Other"

API_ERROR_TYPES = {
    "AttributeError": ErrorType.ATTRIBUTE_ERROR,
    "NameError": ErrorType.NAME_ERROR,
    "TypeError": ErrorType.TYPE_ERROR,
    "AssertionError": ErrorType.ASSERTION_ERROR,
    "ImportError": ErrorType.IMPORT_ERROR,
    "SyntaxError": ErrorType.SYNTAX_ERROR,
    "ValueError": ErrorType.VALUE_ERROR,
    "IndexError": ErrorType.INDEX_ERROR,
    "KeyError": ErrorType.KEY_ERROR,
    "RuntimeError": ErrorType.RUNTIME_ERROR
}

@dataclass
class ErrorInfo:
    type: ErrorType
    message: str
    context: str = ""
    line_number: Optional[int] = None
    file_name: Optional[str] = None

class TestErrorAnalyzer:
    def __init__(self, available_apis: List[str], repo_context: Dict = None):
        """
        Initialize the test error analyzer
        
        Args:
            available_apis: List of available API signatures
            repo_context: Additional context about the repository (classes, methods etc.)
        """
        self.available_apis = available_apis
        self.repo_context = repo_context or {}
        
    def parse_pytest_output(self, output: str) -> List[ErrorInfo]:
        """Parse pytest output to extract error information"""
        errors = []
        
        # Match error patterns with line numbers and files
        error_pattern = r"(?:^|.*?)\b((?:Attribute|Name|Type|Assertion|Import|Syntax|Value|Index|Key|Runtime)Error):?\s*(.+?)(?=\n\n|\Z)"
        line_pattern = r".*line\s+(\d+).*"
        file_pattern = r"([\w\/\\]+\.py)"
        
        matches = re.finditer(error_pattern, output, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            error_type = match.group(1)
            message = match.group(2).strip()
            
            # Extract line number and file if available
            line_match = re.search(line_pattern, message)
            file_match = re.search(file_pattern, message)
            
            line_number = int(line_match.group(1)) if line_match else None
            file_name = file_match.group(1) if file_match else None
            
            errors.append(ErrorInfo(
                ErrorType(error_type),
                message,
                line_number=line_number,
                file_name=file_name
            ))
            
        return errors

    def generate_feedback_prompt(self, error: ErrorInfo) -> str:
        """Generate appropriate feedback prompt for the error"""
        base_prompts = {
            ErrorType.ATTRIBUTE_ERROR: self._handle_attribute_error,
            ErrorType.NAME_ERROR: self._handle_name_error,  
            ErrorType.TYPE_ERROR: self._handle_type_error,
            ErrorType.ASSERTION_ERROR: self._handle_assertion_error,
            ErrorType.IMPORT_ERROR: self._handle_import_error,
            ErrorType.SYNTAX_ERROR: self._handle_syntax_error,
            ErrorType.VALUE_ERROR: self._handle_value_error,
            ErrorType.INDEX_ERROR: self._handle_index_error,
            ErrorType.KEY_ERROR: self._handle_key_error,
            ErrorType.RUNTIME_ERROR: self._handle_runtime_error
        }
        
        handler = base_prompts.get(error.type, self._handle_other_error)
        return handler(error)

    def _handle_attribute_error(self, error: ErrorInfo) -> str:
        close_matches = self._find_similar_apis(error.message)
        line_info = f" at line {error.line_number}" if error.line_number else ""
        file_info = f" in file {error.file_name}" if error.file_name else ""
        
        return f"""You are tasked with generating test cases for a given github issue. 
The code with golden patch should pass the testcase while fail without golden patch. 
Now, the testcase **failed** even **after** applying the patch. You should improve it.
It seems lile you have attempted to use a non-existent API or attribute{line_info}{file_info}.

GitHub issue description:

Error Details:
{error.message}

Patch:

Project Tree:

Available Related APIs:
{close_matches}

Please modify your test case following these guidelines:

1. API Usage:
   - For the current repo only use APIs that are explicitly documented
   - Verify API signatures and return types
   - Check object initialization before method calls

2. Common Attribute Error Causes:
   - Misspelled method or attribute names
   - Using methods from wrong object type
   - Accessing attributes before initialization
   - Confusion between instance and class methods

3. Best Practices:
   - Review the API documentation thoroughly
   - Use code completion tools when available
   - Verify object types before method calls
   - Add type hints to catch errors early

4. Repository Context:
   - Ensure imports are correct
   - Check class inheritance hierarchies
   - Verify API version compatibility

Please regenerate the test case with correct API usage and proper attribute access patterns.

Sample corrected usage:
```python
# Instead of:
result = api.create_users(...)  # Wrong

# Use:
result = api.create_user(...)   # Correct
```
"""

    def _handle_name_error(self, error: ErrorInfo) -> str:
        line_info = f" at line {error.line_number}" if error.line_number else ""
        
        return f"""A Name Error occurred{line_info}, indicating use of undefined variables or functions.

Error Details:
{error.message}

Please address the following aspects:

1. Variable Scoping:
   - Ensure variables are defined before use
   - Check variable scope (global vs local)
   - Verify import statements are at module level

2. Common Causes:
   - Misspelled variable names
   - Missing variable definitions
   - Incorrect import statements
   - Using variables outside their scope
   - Case sensitivity issues

3. Best Practices:
   - Initialize all variables before use
   - Use meaningful variable names
   - Add type hints for better clarity
   - Consider using linters to catch undefined names

4. Code Structure:
   - Review function parameters
   - Check class attribute definitions
   - Verify fixture definitions in pytest
   - Ensure proper test setup

Please regenerate the test case ensuring all names are properly defined and in scope.

Sample corrected pattern:
```python
# Instead of:
def test_function():
    result = undefined_variable  # Wrong

# Use:
def test_function():
    defined_variable = setup_value()
    result = defined_variable    # Correct
```"""

    def _handle_type_error(self, error: ErrorInfo) -> str:
        line_info = f" at line {error.line_number}" if error.line_number else ""
        
        return f"""A Type Error occurred{line_info}, indicating incompatible type operations.

Error Details:
{error.message}

Please review and address:

1. Type Compatibility:
   - Check parameter types match function signatures
   - Verify return type handling
   - Ensure proper type conversions
   - Review collection type operations

2. Common Type Error Scenarios:
   - Mixing incompatible types in operations
   - Incorrect function argument types
   - Invalid type conversions
   - Collection type mismatches
   - None type operations

3. Best Practices:
   - Use type hints to prevent type errors
   - Add explicit type conversions where needed
   - Validate input types early
   - Handle None cases explicitly
   - Use isinstance() for type checking

4. Testing Considerations:
   - Test edge cases with different types
   - Include type boundary tests
   - Verify type conversions
   - Test None handling

Please regenerate the test case with proper type handling and validation.

Sample correct type handling:
```python
# Instead of:
def test_function():
    value = "123"
    result = math_operation(value)  # Wrong

# Use:
def test_function():
    value = int("123")
    result = math_operation(value)  # Correct
```"""

    def _handle_assertion_error(self, error: ErrorInfo) -> str:
        return f"""An Assertion Error occurred in your test case, indicating a failed assertion.

Error Details:
{error.message}

Please analyze and revise:

1. Assertion Analysis:
   - Review expected vs actual values
   - Check assertion logic
   - Verify test prerequisites
   - Consider boundary conditions

2. Common Assertion Failures:
   - Incorrect expected values
   - Floating-point comparison issues
   - Object equality vs identity
   - Sequence order mismatches
   - Time-dependent failures

3. Testing Best Practices:
   - Use descriptive assertion messages
   - Test edge cases explicitly
   - Consider data dependencies
   - Handle asynchronous operations
   - Include positive and negative tests

4. Advanced Testing Patterns:
   - Parameterized testing
   - Fixture usage
   - Setup and teardown
   - Mock object behavior
   - Exception testing

Please regenerate the test case with proper assertions and test conditions.

Sample assertion patterns:
```python
# Instead of:
def test_function():
    assert result == expected  # Basic

# Use:
def test_function():
    assert result == expected, f"Expected {{EXPECTED MESSAGE}}, but got {{RESULT}}"
    # Or use pytest.approx for floats
    assert value == pytest.approx(expected_float, rel=1e-6)
```"""

    def _find_similar_apis(self, error_msg: str) -> str:
        """Find similar APIs based on the error message"""
        # Extract the attempted API call from error message
        api_match = re.search(r"no attribute '(\w+)'", error_msg)
        if api_match:
            attempted_api = api_match.group(1)
            close_matches = get_close_matches(attempted_api, 
                                           [api.split('(')[0] for api in self.available_apis],
                                           n=3,
                                           cutoff=0.6)
            
            relevant_apis = []
            for match in close_matches:
                relevant_apis.extend([api for api in self.available_apis if api.startswith(match)])
                
            if relevant_apis:
                return "Suggested APIs:\n" + "\n".join(f"- {api}" for api in relevant_apis)
            
        return "Available APIs:\n" + "\n".join(f"- {api}" for api in self.available_apis)

    def _handle_import_error(self, error: ErrorInfo) -> str:
        return """An Import Error occurred while trying to import a module or object.

Detailed Analysis Required:

1. Module Dependencies:
   - Verify package installation
   - Check import path correctness
   - Review package versions
   - Inspect virtual environment

2. Common Import Issues:
   - Missing dependencies
   - Incorrect import paths
   - Circular imports
   - Version conflicts
   - Name collisions

3. Resolution Steps:
   - Check requirements.txt
   - Verify PYTHONPATH
   - Review project structure
   - Check package compatibility
   - Use absolute imports

4. Testing Considerations:
   - Mock external dependencies
   - Use dependency injection
   - Handle optional imports
   - Test environment isolation

Please regenerate the test case with proper import handling."""
        
def extract_error_type(error_msg):
    """
    Extract the error type from the error message.
    
    Args:
    error_msg (str): The error message string.
    
    Returns:
    str: The error type extracted from the error message.
    """
    error_type = re.search(r"(?:^|.*?)\b((?:Attribute|Name|Type|Assertion|Import|Syntax|Value|Index|Key|Runtime)Error):?\s*(.+?)(?=\n\n|\Z)", error_msg, re.MULTILINE | re.DOTALL)
    return error_type.group(1) if error_type else "Other"

def test_extract_error_type():
    error_msg = """
    def test_user_creation():
        >       result = api.create_users("test_user", "
        E       AttributeError: module 'api' has no attribute 'create_users'
    """
    assert extract_error_type(error_msg) == "AttributeError"
    print("Test passed.")
    
if __name__ == "__main__":
    test_extract_error_type()
    
# # Usage Example
# if __name__ == "__main__":
#     available_apis = [
#         "api.create_user(username: str, email: str) -> User",
#         "api.delete_user(user_id: int) -> bool",
#         "api.update_user(user_id: int, **kwargs) -> User",
#         "api.get_user(user_id: int) -> Optional[User]"
#     ]
    
#     analyzer = TestErrorAnalyzer(available_apis)
    
#     sample_output = """
#     def test_user_creation():
#         >       result = api.create_users("test_user", "test@example.com")
#         E       AttributeError: module 'api' has no attribute 'create_users'
#     """
    
#     errors = analyzer.parse_pytest_output(sample_output)
#     for error in errors:
#         feedback_prompt = analyzer.generate_feedback_prompt(error)
#         print(f"\n{feedback_prompt}")