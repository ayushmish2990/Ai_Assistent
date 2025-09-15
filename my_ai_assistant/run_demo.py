#!/usr/bin/env python3
"""Demo script to run the AI Assistant and test its capabilities."""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import the my_ai_assistant package
sys.path.insert(0, str(Path(__file__).parent.parent))

from my_ai_assistant import AIAssistant, Config, CapabilityType

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 50)
    print(f" {title} ")
    print("=" * 50)

def main():
    """Run a demonstration of the AI Assistant."""
    # Check if the OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your API key with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print_separator("Initializing AI Assistant")
    
    # Initialize the assistant with a custom configuration
    config = Config()
    config.model.model_name = "gpt-4"  # or any other model you want to use
    config.model.temperature = 0.7
    config.debug = True
    config.log_level = "INFO"
    
    print(f"Using model: {config.model.model_name}")
    print(f"Temperature: {config.model.temperature}")
    print(f"Debug mode: {config.debug}")
    
    assistant = AIAssistant(config)
    
    # Demo 1: Generate Python code
    print_separator("Demo 1: Generate Python Code")
    prompt = "Create a function that checks if a number is prime"
    print(f"Prompt: {prompt}")
    print("Generating code...")
    
    start_time = time.time()
    result = assistant.generate_code(prompt=prompt, language="python")
    elapsed = time.time() - start_time
    
    print(f"\nGenerated code (in {elapsed:.2f} seconds):\n")
    print(result["code"])
    
    # Demo 2: Generate JavaScript code
    print_separator("Demo 2: Generate JavaScript Code")
    prompt = "Create a function that sorts an array of objects by a specific property"
    print(f"Prompt: {prompt}")
    print("Generating code...")
    
    start_time = time.time()
    result = assistant.generate_code(prompt=prompt, language="javascript")
    elapsed = time.time() - start_time
    
    print(f"\nGenerated code (in {elapsed:.2f} seconds):\n")
    print(result["code"])
    
    # Demo 3: Fix buggy code
    print_separator("Demo 3: Fix Buggy Code")
    buggy_code = """
    def calculate_average(numbers):
        total = 0
        for num in numbers:
            total += num
        return total / len(numbers)  # This will raise ZeroDivisionError if numbers is empty
    """
    print(f"Buggy code:\n{buggy_code}")
    print("Fixing code...")
    
    start_time = time.time()
    result = assistant.fix_code(
        code=buggy_code,
        error_message="ZeroDivisionError: division by zero",
        language="python"
    )
    elapsed = time.time() - start_time
    
    print(f"\nFixed code (in {elapsed:.2f} seconds):\n")
    print(result["fixed_code"])
    print(f"\nExplanation:\n{result['explanation']}")
    
    # Print usage statistics
    print_separator("Usage Statistics")
    for stat_name, stat_value in assistant.stats.items():
        print(f"{stat_name}: {stat_value}")

if __name__ == "__main__":
    main()