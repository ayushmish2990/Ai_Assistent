#!/usr/bin/env python3
"""
Multi-Provider AI Assistant Demo

This script demonstrates how to use the AI assistant with different providers:
- OpenAI (GPT models)
- Hugging Face (Open-source models)
- Anthropic (Claude models)
- Local models

Make sure to set your API keys as environment variables before running:
- OPENAI_API_KEY
- HUGGINGFACE_API_KEY (optional for public models)
- ANTHROPIC_API_KEY
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.config import Config
from core.model_manager import ModelManager
from core.main import AICodingAssistant

def load_config():
    """Load configuration from the multi-provider config file."""
    config_path = Path(__file__).parent / "config_multi_provider.yaml"
    if config_path.exists():
        return Config.from_yaml(str(config_path))
    else:
        print(f"Configuration file not found: {config_path}")
        print("Using default configuration...")
        return Config()

def check_api_keys():
    """Check which API keys are available."""
    available_providers = []
    
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append("OpenAI")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append("Anthropic")
    
    # Hugging Face can work without API key for public models
    available_providers.append("Hugging Face")
    available_providers.append("Local Models")
    
    return available_providers

def demo_code_generation(assistant, provider=None):
    """Demonstrate code generation capabilities."""
    print(f"\n{'='*60}")
    print(f"CODE GENERATION DEMO{' - ' + provider.upper() if provider else ''}")
    print(f"{'='*60}")
    
    prompt = "Create a Python function that calculates the factorial of a number using recursion."
    
    try:
        if provider:
            response = assistant.model_manager.generate(prompt, provider=provider)
        else:
            response = assistant.model_manager.generate(prompt)
        
        print(f"Prompt: {prompt}")
        print(f"\nResponse:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
    except Exception as e:
        print(f"Error generating code: {e}")

def demo_code_explanation(assistant, provider=None):
    """Demonstrate code explanation capabilities."""
    print(f"\n{'='*60}")
    print(f"CODE EXPLANATION DEMO{' - ' + provider.upper() if provider else ''}")
    print(f"{'='*60}")
    
    code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
    
    prompt = f"Explain how this Python code works:\n\n{code}"
    
    try:
        if provider:
            response = assistant.model_manager.generate(prompt, provider=provider)
        else:
            response = assistant.model_manager.generate(prompt)
        
        print(f"Code to explain:")
        print(code)
        print(f"\nExplanation:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
    except Exception as e:
        print(f"Error explaining code: {e}")

def demo_debugging_help(assistant, provider=None):
    """Demonstrate debugging assistance."""
    print(f"\n{'='*60}")
    print(f"DEBUGGING DEMO{' - ' + provider.upper() if provider else ''}")
    print(f"{'='*60}")
    
    buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # This will cause ZeroDivisionError if numbers is empty

# Test
result = calculate_average([])
print(result)
"""
    
    prompt = f"Find and fix the bug in this Python code:\n\n{buggy_code}"
    
    try:
        if provider:
            response = assistant.model_manager.generate(prompt, provider=provider)
        else:
            response = assistant.model_manager.generate(prompt)
        
        print(f"Buggy code:")
        print(buggy_code)
        print(f"\nDebugging help:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
    except Exception as e:
        print(f"Error debugging code: {e}")

def demo_provider_comparison(assistant):
    """Compare responses from different providers."""
    print(f"\n{'='*60}")
    print("PROVIDER COMPARISON DEMO")
    print(f"{'='*60}")
    
    prompt = "Write a Python function to check if a string is a palindrome."
    providers = ["openai", "huggingface", "anthropic", "local"]
    
    for provider in providers:
        try:
            print(f"\n--- {provider.upper()} ---")
            response = assistant.model_manager.generate(prompt, provider=provider, max_tokens=300)
            print(response[:200] + "..." if len(response) > 200 else response)
        except Exception as e:
            print(f"Error with {provider}: {e}")

def main():
    """Main demo function."""
    print("Multi-Provider AI Assistant Demo")
    print("=" * 40)
    
    # Check available providers
    available_providers = check_api_keys()
    print(f"Available providers: {', '.join(available_providers)}")
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config()
    
    # Initialize AI Assistant
    print("Initializing AI Assistant...")
    try:
        assistant = AICodingAssistant(config)
        print("AI Assistant initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize AI Assistant: {e}")
        return
    
    # Run demos with default provider
    print(f"\nRunning demos with default provider ({config.providers.default_provider})...")
    demo_code_generation(assistant)
    demo_code_explanation(assistant)
    demo_debugging_help(assistant)
    
    # Run provider comparison if multiple providers are available
    if len(available_providers) > 1:
        demo_provider_comparison(assistant)
    
    # Demo specific providers
    if "OpenAI" in available_providers:
        demo_code_generation(assistant, "openai")
    
    if "Anthropic" in available_providers:
        demo_code_explanation(assistant, "anthropic")
    
    # Always try Hugging Face and Local (they don't require API keys)
    demo_debugging_help(assistant, "huggingface")
    demo_code_generation(assistant, "local")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    
    # Usage statistics
    print("\nUsage Tips:")
    print("- Set OPENAI_API_KEY for OpenAI models")
    print("- Set ANTHROPIC_API_KEY for Claude models")
    print("- Set HUGGINGFACE_API_KEY for private Hugging Face models (optional for public)")
    print("- Local models run on your hardware (no API key needed)")
    print("- Modify config_multi_provider.yaml to customize settings")
    print("- Use assistant.model_manager.generate(prompt, provider='provider_name') for specific providers")

if __name__ == "__main__":
    main()