# My AI Assistant Usage Guide

This guide provides detailed instructions on how to use the My AI Assistant for various code-related tasks with support for multiple AI providers.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Basic Usage](#basic-usage)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- At least one AI provider API key (OpenAI, Anthropic, or Hugging Face)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/my-ai-assistant.git
cd my-ai-assistant

# Install dependencies
pip install -r my_ai_assistant/requirements.txt
```

### Environment Setup

Set up your API keys for the providers you want to use:

```bash
# On Linux/macOS
# For OpenAI (GPT models)
export OPENAI_API_KEY="your-openai-api-key-here"

# For Anthropic (Claude models)
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"

# For Hugging Face (optional for public models)
export HUGGINGFACE_API_KEY="your-huggingface-token-here"

# On Windows (PowerShell)
$env:OPENAI_API_KEY="your-openai-api-key-here"
$env:ANTHROPIC_API_KEY="your-anthropic-api-key-here"
$env:HUGGINGFACE_API_KEY="your-huggingface-token-here"
```

## Configuration

The assistant can be configured using a YAML file or by directly modifying the `Config` object. The new multi-provider configuration allows you to use multiple AI providers simultaneously.

### Single Provider Configuration (config.yaml)

Create a `config.yaml` file with the following structure:

```yaml
model:
  model_name: "gpt-4"  # OpenAI model to use
  max_tokens: 2048     # Maximum tokens for completion
  temperature: 0.7     # Creativity level (0.0-1.0)
  top_p: 1.0           # Nucleus sampling parameter
  frequency_penalty: 0.0  # Penalty for token frequency
  presence_penalty: 0.0   # Penalty for token presence

# General settings
debug: false           # Enable debug mode
log_level: "INFO"      # Logging level (DEBUG, INFO, WARNING, ERROR)

# API server settings
api:
  host: "0.0.0.0"      # Host to bind the API server
  port: 8000           # Port for the API server
  reload: true         # Enable auto-reload for development
```

### Multi-Provider Configuration (config_multi_provider.yaml)

For advanced usage with multiple AI providers:

```yaml
providers:
  openai:
    model_name: "gpt-3.5-turbo"
    api_key: "${OPENAI_API_KEY}"
    max_tokens: 1000
    temperature: 0.7
  
  anthropic:
    model_name: "claude-3-sonnet-20240229"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1000
    temperature: 0.7
  
  huggingface:
    model_name: "microsoft/DialoGPT-medium"
    api_key: "${HUGGINGFACE_API_KEY}"
    max_tokens: 800
    temperature: 0.8
  
  local:
    model_name: "microsoft/DialoGPT-small"
    max_tokens: 500
    temperature: 0.7
  
  default_provider: "openai"

# General settings
debug: false
log_level: "INFO"

# API server settings
api:
  host: "0.0.0.0"
  port: 8000
  reload: true
```

### Programmatic Configuration

```python
from my_ai_assistant import Config, AIAssistant

config = Config()
config.model.model_name = "gpt-4"
config.model.temperature = 0.7
config.debug = True

assistant = AIAssistant(config)
```

## Basic Usage

### Code Generation

```python
from my_ai_assistant import AIAssistant, Config

# Initialize the assistant with multi-provider support
config = Config.from_yaml("config_multi_provider.yaml")
assistant = AIAssistant(config)

# Generate Python code with default provider
result = assistant.generate_code(
    prompt="Create a function that calculates the Fibonacci sequence",
    language="python"
)
print(result["code"])

# Generate code with specific provider
result = assistant.model_manager.generate(
    "Create a function to sort an array of objects by property",
    provider="openai",
    language="javascript"
)
print(result)

# Use Anthropic Claude for detailed explanations
result = assistant.model_manager.generate(
    "Explain how the quicksort algorithm works with examples",
    provider="anthropic"
)
print(result)
```

### Available Providers

The AI Assistant supports multiple AI providers:

- **OpenAI**: GPT-3.5-turbo, GPT-4, GPT-4-turbo
  - Requires: `OPENAI_API_KEY`
  - Best for: General coding, complex reasoning

- **Anthropic**: Claude-3 (Sonnet, Haiku, Opus)
  - Requires: `ANTHROPIC_API_KEY`
  - Best for: Code analysis, detailed explanations

- **Hugging Face**: Open-source models (DialoGPT, BlenderBot, etc.)
  - Optional: `HUGGINGFACE_API_KEY` (for private models)
  - Best for: Free access, specialized models

- **Local Models**: Run models on your hardware
  - Requires: No API key, local compute resources
  - Best for: Privacy, offline usage, cost control

### Code Fixing

```python
from my_ai_assistant import AIAssistant, Config

# Initialize the assistant
config = Config()
assistant = AIAssistant(config)

# Fix buggy code
buggy_code = """
def divide(a, b):
    return a / b  # This will raise ZeroDivisionError when b is 0
"""

result = assistant.fix_code(
    code=buggy_code,
    error_message="ZeroDivisionError: division by zero",
    language="python"
)

print("Fixed code:")
print(result["fixed_code"])

print("Explanation:")
print(result["explanation"])
```

### Using Specific Capabilities

```python
from my_ai_assistant import AIAssistant, Config, CapabilityType

# Initialize the assistant
config = Config.from_yaml("config_multi_provider.yaml")
assistant = AIAssistant(config)

# Use a specific capability with default provider
result = assistant.execute_capability(
    capability_type=CapabilityType.CODE_GENERATION,
    prompt="Create a function to check if a string is a palindrome",
    language="python"
)
print(result["code"])

# Use specific provider for different tasks
# Use OpenAI for code generation
code_result = assistant.model_manager.generate(
    "Write a Python class for a binary tree",
    provider="openai",
    max_tokens=500
)

# Use Anthropic for code explanation
explanation = assistant.model_manager.generate(
    "Explain the SOLID principles with examples",
    provider="anthropic",
    temperature=0.5
)

# Use Hugging Face for conversational responses
chat_response = assistant.model_manager.generate(
    "Generate a helpful chatbot response",
    provider="huggingface"
)

# Use local model for privacy-sensitive tasks
local_result = assistant.model_manager.generate(
    "Process this sensitive data",
    provider="local"
)
```

## API Reference

### AIAssistant Class

#### `__init__(config=None)`

Initializes the AI Assistant with the given configuration.

- **Parameters:**
  - `config` (Config, optional): Configuration object. If not provided, default configuration is used.

#### `generate_code(prompt, language)`

Generates code based on the given prompt and language.

- **Parameters:**
  - `prompt` (str): Description of the code to generate.
  - `language` (str): Programming language (e.g., "python", "javascript").
- **Returns:**
  - dict: Contains the generated code and metadata.

#### `fix_code(code, error_message, language)`

Fixes issues in the provided code based on the error message.

- **Parameters:**
  - `code` (str): Code with issues to fix.
  - `error_message` (str): Error message or description of the issue.
  - `language` (str): Programming language of the code.
- **Returns:**
  - dict: Contains the fixed code, explanation, and metadata.

#### `execute_capability(capability_type, **kwargs)`

Executes a specific AI capability with the given parameters.

- **Parameters:**
  - `capability_type` (CapabilityType): Type of capability to execute.
  - `**kwargs`: Additional parameters specific to the capability.
- **Returns:**
  - dict: Result of the capability execution.

### Config Class

#### `__init__()`

Initializes a configuration object with default values.

#### `from_yaml(file_path)`

Loads configuration from a YAML file.

- **Parameters:**
  - `file_path` (str): Path to the YAML configuration file.
- **Returns:**
  - Config: Configuration object with values from the file.

#### `to_yaml(file_path)`

Saves the configuration to a YAML file.

- **Parameters:**
  - `file_path` (str): Path where to save the configuration.

## Examples

See the `examples` directory for more usage examples:

- `my_assistant_usage.py`: Basic usage examples
- `run_demo.py`: Interactive demonstration of capabilities

## Troubleshooting

### Common Issues

#### API Key Not Found

```
Error: OPENAI_API_KEY environment variable is not set.
```

**Solution**: Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### Model Not Available

```
Error: The model 'gpt-4' does not exist or you do not have access to it.
```

**Solution**: Change the model in your configuration to one you have access to, such as "gpt-3.5-turbo".

#### Rate Limit Exceeded

```
Error: Rate limit exceeded. Please try again later.
```

**Solution**: Implement retry logic with exponential backoff or reduce the frequency of API calls.

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the logs for detailed error messages
2. Review the OpenAI API documentation
3. Open an issue on the GitHub repository