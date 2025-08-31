"""Model management for the AI Coding Assistant."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    """Handles model loading and text generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model and tokenizer with enhanced error handling."""
        try:
            logger.info(f"Loading tokenizer for model: {self.config.model.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name,
                padding_side="left",
                trust_remote_code=True,
            )
            
            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set model max length if not already set
            if not hasattr(self.tokenizer, 'model_max_length') or self.tokenizer.model_max_length > 4096:
                self.tokenizer.model_max_length = 2048
            
            logger.info(f"Loading model: {self.config.model.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Reduce memory usage during loading
            )
            
            # Move model to device if not using device_map
            if not torch.cuda.is_available() and not hasattr(self.model, 'device'):
                self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            logger.info(f"Model loaded successfully on device: {self.device}")
                
        except Exception as e:
            logger.error(f"Error loading model {self.config.model.model_name}: {str(e)}")
            # Try to load a fallback model if the primary one fails
            if self.config.model.model_name != "gpt2":
                logger.warning("Falling back to gpt2 model")
                self.config.model.model_name = "gpt2"
                self._load_model()
            else:
                raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt with enhanced error handling and input validation."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Please check model initialization.")
        
        try:
            # Validate input
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt must be a non-empty string")
            
            # Truncate prompt if it's too long
            max_prompt_length = self.tokenizer.model_max_length - (max_tokens or self.config.model.max_tokens or 100)
            if len(prompt) > max_prompt_length:
                logger.warning(f"Prompt too long, truncating to {max_prompt_length} characters")
                prompt = prompt[:max_prompt_length]
            
            logger.debug(f"Generating response for prompt: {prompt[:100]}...")
            
            # Tokenize with proper error handling
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                ).to(self.device)
            except Exception as e:
                raise ValueError(f"Failed to tokenize input: {str(e)}")
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                max_new_tokens=min(max_tokens or self.config.model.max_tokens, 512),  # Cap max tokens
                temperature=min(max(temperature or self.config.model.temperature, 0.1), 1.0),  # Clamp temperature
                top_p=min(max(kwargs.pop('top_p', self.config.model.top_p), 0.1), 1.0),  # Clamp top_p
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Generate with error handling
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                    )
                
                # Decode the generated tokens
                generated = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True,
                )
                
                # Remove the input prompt from the generated text if present
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()
                
                logger.debug(f"Generated response: {generated[:200]}...")
                return generated
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("CUDA out of memory. Try reducing max_tokens or using a smaller model.")
                raise
                
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            # Return a helpful error message instead of crashing
            return f"Error generating response: {str(e)}"
