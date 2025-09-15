#!/usr/bin/env python3
"""Script to run the AI Assistant API server."""

import os
import sys
import uvicorn
from pathlib import Path

# Add the parent directory to the path so we can import the my_ai_assistant package
sys.path.insert(0, str(Path(__file__).parent.parent))

from my_ai_assistant.api import app

def main():
    """Run the API server."""
    # Check if the OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("The API server will start, but API calls will fail.")
        print("Set your API key with: export OPENAI_API_KEY='your-api-key-here'")
    
    # Run the server
    print("Starting AI Assistant API server...")
    uvicorn.run(
        "my_ai_assistant.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()