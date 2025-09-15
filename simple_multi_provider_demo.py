#!/usr/bin/env python3
"""
Simple Multi-Provider AI Assistant Demo

This script demonstrates the basic multi-provider functionality.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from core.config import Config, ProvidersConfig, ProviderConfig
    from core.multi_provider_model_manager import (
        MultiProviderModelManager, 
        OpenAIProvider, 
        HuggingFaceProvider, 
        AnthropicProvider, 
        LocalModelProvider
    )
    from core.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

def check_api_keys():
    """Check which API keys are available."""
    available_providers = []
    
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append("OpenAI")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append("Anthropic")
    
    # Hugging Face and Local don't require API keys
    available_providers.extend(["Hugging Face", "Local Models"])
    
    return available_providers

def demo_multi_provider():
    """Demonstrate multi-provider functionality."""
    print("Multi-Provider AI Assistant Demo")
    print("=" * 40)
    
    # Check available providers
    available_providers = check_api_keys()
    print(f"Available providers: {', '.join(available_providers)}")
    
    # Create configuration
    config = Config()
    
    # Setup provider configurations
    if os.getenv("OPENAI_API_KEY"):
        config.providers.openai.api_key = os.getenv("OPENAI_API_KEY")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        config.providers.anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if os.getenv("HUGGINGFACE_API_KEY"):
        config.providers.huggingface.api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    # Create multi-provider manager
    manager = MultiProviderModelManager(config)
    
    # List available providers
    providers = manager.get_available_providers()
    print(f"\nActive providers: {providers}")
    
    if not providers:
        print("No providers available. Please set up API keys or install transformers.")
        return
    
    # Set default provider (prefer local for demo)
    if "local" in providers:
        manager.set_default_provider("local")
        print("Using local model as default provider")
    else:
        manager.set_default_provider(providers[0])
        print(f"Using {providers[0]} as default provider")
    
    print(f"\nConfigured providers: {providers}")
    
    # Test generation with different providers
    test_prompt = "Write a simple Python function to add two numbers."
    
    for provider_name in manager.providers.keys():
        print(f"\n{'='*50}")
        print(f"Testing {provider_name.upper()} provider")
        print(f"{'='*50}")
        
        try:
            response = manager.generate(
                prompt=test_prompt,
                provider=provider_name,
                max_tokens=200,
                temperature=0.7
            )
            print(f"Prompt: {test_prompt}")
            print(f"Response: {response[:300]}{'...' if len(response) > 300 else ''}")
        except Exception as e:
            print(f"Error with {provider_name}: {e}")
    
    # Test default provider
    if manager.providers:
        print(f"\n{'='*50}")
        print("Testing DEFAULT provider")
        print(f"{'='*50}")
        
        try:
            response = manager.generate(
                prompt="Explain what a Python decorator is.",
                max_tokens=150
            )
            print(f"Default provider response: {response[:200]}{'...' if len(response) > 200 else ''}")
        except Exception as e:
            print(f"Error with default provider: {e}")
    
    print("\n" + "="*50)
    print("Demo completed!")
    print("="*50)
    
    # Usage tips
    print("\nUsage Tips:")
    print("- Set OPENAI_API_KEY for OpenAI models")
    print("- Set ANTHROPIC_API_KEY for Claude models")
    print("- Set HUGGINGFACE_API_KEY for private Hugging Face models (optional for public)")
    print("- Local models run on your hardware (no API key needed)")
    print("- Each provider has different strengths and use cases")

if __name__ == "__main__":
    demo_multi_provider()