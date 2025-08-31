"""
Basic usage example for the AI Coding Assistant.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core import AICodingAssistant, Config

def main():
    # Initialize the assistant with default config
    assistant = AICodingAssistant()
    
    # Example 1: Generate code from a prompt
    print("\n=== Example 1: Code Generation ===")
    prompt = """
    Create a Python function that calculates the nth Fibonacci number.
    Include type hints and a docstring.
    """
    generated_code = assistant.generate_code(prompt, language="python")
    print(generated_code)
    
    # Example 2: Fix a bug in code
    print("\n=== Example 2: Bug Fixing ===")
    buggy_code = """
    def divide(a, b):
        return a / b  # This will raise ZeroDivisionError when b is 0
    """
    fixed_code = assistant.fix_code(
        buggy_code,
        error_message="Fix the division by zero error",
        language="python"
    )
    print(fixed_code)
    
    # Example 3: Explain code
    print("\n=== Example 3: Code Explanation ===")
    code_to_explain = """
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    """
    explanation = assistant.explain_code(code_to_explain, language="python")
    print(explanation)

if __name__ == "__main__":
    main()
