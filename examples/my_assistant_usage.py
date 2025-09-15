#!/usr/bin/env python3
"""Basic usage example for the My AI Assistant."""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the my_ai_assistant package
sys.path.insert(0, str(Path(__file__).parent.parent))

from my_ai_assistant import AIAssistant, Config, CapabilityType

def main():
    # Make sure to set your OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize the assistant
    config = Config()
    config.model.model_name = "gpt-4"  # or any other model you want to use
    assistant = AIAssistant(config)
    
    # Example 1: Generate code
    print("\n=== Example 1: Generate Code ===")
    result = assistant.generate_code(
        prompt="Create a function that calculates the Fibonacci sequence up to n terms",
        language="python"
    )
    print(f"Generated code:\n{result['code']}")
    
    # Example 2: Fix code with an error
    print("\n=== Example 2: Fix Code ===")
    broken_code = """
def factorial(n):
    if n == 0:
        return 0  # This is incorrect, should be 1
    return n * factorial(n-1)
"""
    result = assistant.fix_code(
        code=broken_code,
        error_message="RecursionError: maximum recursion depth exceeded",
        language="python"
    )
    print(f"Fixed code:\n{result['fixed_code']}")
    print(f"Explanation:\n{result['explanation']}")
    
    # Example 3: Using a specific capability directly
    print("\n=== Example 3: Using a Specific Capability ===")
    result = assistant.execute_capability(
        capability_type=CapabilityType.CODE_GENERATION,
        prompt="Create a function to check if a string is a palindrome",
        language="python"
    )
    print(f"Generated code:\n{result['code']}")
    
    # Print usage statistics
    print("\n=== Usage Statistics ===")
    for stat_name, stat_value in assistant.stats.items():
        print(f"{stat_name}: {stat_value}")

if __name__ == "__main__":
    main()