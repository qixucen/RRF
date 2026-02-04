"""Code answer extraction tool"""

import re
import subprocess
import tempfile
import os
from api.llm import gen

CODE_PROMPT_SUFFIX = "\n\nSolve this step by step. Put your final code solution in a code block with ```python and ```."


def extract_code_block(text: str, language: str = "python", fallback_any: bool = True) -> str | None:
    """
    Extract code block from LLM output
    
    Args:
        text: LLM output text
        language: Target language (e.g., python, javascript)
        fallback_any: If no language-specific block found, fallback to any code block
    
    Returns:
        Extracted code string, or None on failure
    """
    if not text:
        return None
    
    # First try to extract language-specific code block ```python ... ```
    pattern = rf'```{language}\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    
    # Fallback: extract any code block ``` ... ```
    if fallback_any:
        pattern = r'```(?:\w*)\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
    
    return None


def extract_all_code_blocks(text: str, language: str = None) -> list[str]:
    """
    Extract all code blocks from LLM output
    
    Args:
        text: LLM output text
        language: Optional, filter by language
    
    Returns:
        List of code blocks
    """
    if not text:
        return []
    
    if language:
        pattern = rf'```{language}\s*\n(.*?)```'
    else:
        pattern = r'```(?:\w*)\s*\n(.*?)```'
    
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches]


def extract_function(text: str, function_name: str = None) -> str | None:
    """
    Extract function definition from code
    
    Args:
        text: Code text
        function_name: Optional, specific function name
    
    Returns:
        Function code, or None on failure
    """
    if not text:
        return None
    
    # First try to extract from code block
    code = extract_code_block(text)
    if code:
        text = code
    
    if function_name:
        # Extract specific function
        pattern = rf'(def\s+{function_name}\s*\(.*?\):.*?)(?=\ndef\s|\nclass\s|\Z)'
    else:
        # Extract last function
        pattern = r'(def\s+\w+\s*\(.*?\):.*?)(?=\ndef\s|\nclass\s|\Z)'
    
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    return None


def extract_class(text: str, class_name: str = None) -> str | None:
    """
    Extract class definition from code
    
    Args:
        text: Code text
        class_name: Optional, specific class name
    
    Returns:
        Class code, or None on failure
    """
    if not text:
        return None
    
    # First try to extract from code block
    code = extract_code_block(text)
    if code:
        text = code
    
    if class_name:
        pattern = rf'(class\s+{class_name}\s*(?:\(.*?\))?:.*?)(?=\nclass\s|\Z)'
    else:
        pattern = r'(class\s+\w+\s*(?:\(.*?\))?:.*?)(?=\nclass\s|\Z)'
    
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    return None


def execute_code(code: str, timeout: float = 10.0, capture_output: bool = True) -> tuple[str | None, str | None, int]:
    """
    Execute Python code and return results
    
    Args:
        code: Code to execute
        timeout: Timeout in seconds
        capture_output: Whether to capture output
    
    Returns:
        (stdout, stderr, return_code) tuple
    """
    if not code:
        return None, "No code provided", -1
    
    # Create temp file to execute code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return None, "Execution timeout", -2
    except Exception as e:
        return None, str(e), -3
    finally:
        os.unlink(temp_file)


def validate_syntax(code: str) -> tuple[bool, str | None]:
    """
    Validate code syntax
    
    Args:
        code: Code to validate
    
    Returns:
        (is_valid, error_message) tuple
    """
    if not code:
        return False, "No code provided"
    
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def clean_code(code: str) -> str:
    """
    Clean code by removing extra blank lines
    
    Args:
        code: Original code
    
    Returns:
        Cleaned code
    """
    if not code:
        return ""
    
    lines = code.split('\n')
    
    # Remove leading and trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)


async def solve_code(problem: str, model: str = "gpt-4o-mini", temperature: float = 0.7,
                     add_prompt_suffix: bool = True, language: str = "python",
                     validate: bool = True) -> tuple[str | None, str | None, any]:
    """
    Solve code problem and extract answer
    
    Args:
        problem: Code problem description
        model: Model to use
        temperature: Temperature parameter
        add_prompt_suffix: Whether to auto-add code block suffix
        language: Target programming language
        validate: Whether to validate syntax
    
    Returns:
        (extracted_code, full_response, validation_result) tuple
    """
    suffix = CODE_PROMPT_SUFFIX
    if language != "python":
        suffix = f"\n\nSolve this step by step. Put your final code solution in a code block with ```{language} and ```."
    
    prompt = problem + suffix if add_prompt_suffix else problem
    response, _ = await gen(prompt=prompt, model=model, temperature=temperature)
    
    code = extract_code_block(response, language=language) if response else None
    
    # Syntax validation (Python only)
    validation_result = None
    if validate and code and language == "python":
        is_valid, error = validate_syntax(code)
        validation_result = {"valid": is_valid, "error": error}
    
    return code, response, validation_result


async def solve_and_execute(problem: str, model: str = "gpt-4o-mini", temperature: float = 0.7,
                            timeout: float = 10.0) -> dict:
    """
    Solve code problem, extract code and execute
    
    Args:
        problem: Code problem description
        model: Model to use
        temperature: Temperature parameter
        timeout: Execution timeout
    
    Returns:
        Dictionary containing code, response, and execution result
    """
    code, response, validation = await solve_code(problem, model, temperature)
    
    result = {
        "code": code,
        "response": response,
        "validation": validation,
        "execution": None
    }
    
    if code and validation and validation.get("valid"):
        stdout, stderr, return_code = execute_code(code, timeout=timeout)
        result["execution"] = {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code,
            "success": return_code == 0
        }
    
    return result
