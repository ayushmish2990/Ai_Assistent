import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core import Config, AICodingAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ai_assistant():
    """Test the AI Coding Assistant with basic operations."""
    try:
        # Initialize with test configuration
        config = Config()
        config.model.model_name = "gpt2"  # Use a smaller model for testing
        config.model.max_tokens = 100
        config.model.temperature = 0.1
        
        logger.info("Initializing AI Coding Assistant...")
        assistant = AICodingAssistant(config)
        
        # Test 1: Code Generation
        logger.info("Testing code generation...")
        prompt = "Create a function that returns 'Hello, World!'"
        code = assistant.generate_code(prompt, language="python")
        logger.info(f"Generated code:\n{code}")
        assert isinstance(code, str) and len(code) > 0, "Code generation failed"
        
        # Test 2: Code Explanation
        logger.info("Testing code explanation...")
        explanation = assistant.explain_code("def add(a, b): return a + b", language="python")
        logger.info(f"Code explanation:\n{explanation}")
        assert isinstance(explanation, str) and len(explanation) > 0, "Code explanation failed"
        
        logger.info("All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_ai_assistant()
    sys.exit(0 if success else 1)
