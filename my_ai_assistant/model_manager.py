"""Model management for the AI Assistant."""
import logging
from typing import Any, Dict, Optional

import openai

from .config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    """Handles model loading and text generation."""
    
    def __init__(self, config: Config):
        """Initialize the model manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self._setup_client()
    
    def _setup_client(self) -> None:
        """Set up the OpenAI client."""
        try:
            self.client = openai.OpenAI()
            logger.info(f"OpenAI client initialized for model: {self.config.model.model_name}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise
    
    def generate(self, 
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                **kwargs) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: The prompt to generate text from
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text
        """
        try:
            # Validate input
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt must be a non-empty string")
            
            logger.debug(f"Generating response for prompt: {prompt[:100]}...")
            
            # Set up parameters
            params = {
                "model": self.config.model.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or self.config.model.max_tokens,
                "temperature": temperature or self.config.model.temperature,
                "top_p": self.config.model.top_p,
                "frequency_penalty": self.config.model.frequency_penalty,
                "presence_penalty": self.config.model.presence_penalty,
            }
            
            # Add stop sequences if available
            if self.config.model.stop_sequences:
                params["stop"] = self.config.model.stop_sequences
            
            # Add additional parameters
            params.update({k: v for k, v in kwargs.items() if v is not None})
            
            # Generate response
            response = self.client.chat.completions.create(**params)
            
            # Extract and return text
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise