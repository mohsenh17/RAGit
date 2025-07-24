import os
import ast
from typing import List


def extract_code_elements(file_path: str) -> List[str]:
    """
    Extracts all functions and classes from a Python file,
    including their docstrings and full source code.

    Args:
        file_path (str): Path to a Python file.

    Returns:
        List[str]: A list of formatted code snippets.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        # Syntax errors
        return []

    code_elements = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            docstring = ast.get_docstring(node) or "No docstring available."
            snippet = ast.unparse(node)
            element_type = "Class" if isinstance(node, ast.ClassDef) else "Function"
            formatted_block = (
                f"File: {os.path.basename(file_path)}\n"
                f"[{element_type}] {name}:\n"
                f"Docstring: {docstring}\n"
                f"Code:\n{snippet}"
            )
            code_elements.append(formatted_block)

    return code_elements


def parse_repo(repo_path: str) -> List[str]:
    """
    Walks through a repository and extracts all functions and classes
    from Python files.

    Args:
        repo_path (str): Path to the repository.

    Returns:
        List[str]: A list of formatted code snippets from the repo.
    """
    all_snippets = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                elements = extract_code_elements(file_path)
                all_snippets.extend(elements)
    return all_snippets


if __name__ == "__main__":
    # Example usage for testing
    snippets = parse_repo("./examples/aminoClust")
    for snippet in snippets:
        print("-----\n", snippet)
