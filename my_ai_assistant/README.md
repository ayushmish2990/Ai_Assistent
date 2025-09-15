# My AI Assistant

A powerful AI assistant for code generation, debugging, and other software development tasks.

## Features

- **Code Generation**: Generate high-quality code from natural language descriptions
- **Debugging**: Automatically detect and fix bugs in code
- **API Server**: Expose AI capabilities through a RESTful API

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/my-ai-assistant.git
   cd my-ai-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Basic Usage

```python
from my_ai_assistant import AIAssistant, Config

# Initialize the assistant
config = Config()
assistant = AIAssistant(config)

# Generate code
result = assistant.generate_code(
    prompt="Create a function that calculates the Fibonacci sequence",
    language="python"
)
print(result["code"])

# Fix code
result = assistant.fix_code(
    code="def factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n-1)",
    error_message="RecursionError: maximum recursion depth exceeded",
    language="python"
)
print(result["fixed_code"])
```

### Running the API Server

```bash
python -m my_ai_assistant.api
```

The API server will be available at `http://localhost:8000`.

## API Endpoints

- `POST /generate`: Generate code from a natural language prompt
- `POST /fix`: Fix issues in code
- `POST /analyze`: Analyze code for errors
- `GET /health`: Health check endpoint

## Configuration

You can customize the assistant by creating a configuration file:

```yaml
model:
  model_name: gpt-4
  max_tokens: 2048
  temperature: 0.7
debug: false
log_level: INFO
```

Then load it in your code:

```python
from my_ai_assistant import AIAssistant, Config

config = Config.from_yaml("config.yaml")
assistant = AIAssistant(config)
```

## Examples

See the `examples` directory for more usage examples.

## License

MIT