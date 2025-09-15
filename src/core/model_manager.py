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
from .multi_provider_model_manager import MultiProviderModelManager, OpenAIProvider, HuggingFaceProvider, AnthropicProvider, LocalModelProvider

logger = logging.getLogger(__name__)

class ModelManager:
    """Handles model loading and text generation with multi-provider support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize multi-provider manager
        self.multi_provider_manager = MultiProviderModelManager()
        self._setup_providers()
        
        # Legacy support - keep original model/tokenizer for backward compatibility
        self.model = None
        self.tokenizer = None
        if hasattr(config, 'model') and config.model.model_name:
            self._load_model()
    
    def _setup_providers(self):
        """Setup all available providers based on configuration."""
        providers_config = self.config.providers
        
        # Setup OpenAI provider
        if providers_config.openai and providers_config.openai.api_key:
            openai_provider = OpenAIProvider(providers_config.openai.api_key)
            self.multi_provider_manager.add_provider("openai", openai_provider)
            logger.info("OpenAI provider configured")
        
        # Setup Hugging Face provider
        if providers_config.huggingface:
            hf_provider = HuggingFaceProvider(
                providers_config.huggingface.api_key,
                providers_config.huggingface.model_name
            )
            self.multi_provider_manager.add_provider("huggingface", hf_provider)
            logger.info("Hugging Face provider configured")
        
        # Setup Anthropic provider
        if providers_config.anthropic and providers_config.anthropic.api_key:
            anthropic_provider = AnthropicProvider(providers_config.anthropic.api_key)
            self.multi_provider_manager.add_provider("anthropic", anthropic_provider)
            logger.info("Anthropic provider configured")
        
        # Setup Local provider
        if providers_config.local:
            local_provider = LocalModelProvider(providers_config.local.model_name)
            self.multi_provider_manager.add_provider("local", local_provider)
            logger.info("Local provider configured")
        
        # Set default provider
        if providers_config.default_provider in self.multi_provider_manager.providers:
            self.multi_provider_manager.set_default_provider(providers_config.default_provider)
            logger.info(f"Default provider set to: {providers_config.default_provider}")
    
    def _load_model(self) -> None:
        """Load model and tokenizer with enhanced error handling (legacy support)."""
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
                logger.warning("Legacy model loading failed, using multi-provider system only")
    
    def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt using multi-provider system with legacy fallback."""
        try:
            # Use multi-provider system first
            if provider:
                response = self.multi_provider_manager.generate(prompt, provider, max_tokens=max_tokens, temperature=temperature, **kwargs)
            else:
                response = self.multi_provider_manager.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
            
            return response
            
        except Exception as e:
            logger.warning(f"Multi-provider generation failed: {e}")
            
            # Fallback to legacy model if available
            if self.model is not None and self.tokenizer is not None:
                logger.info("Falling back to legacy model")
                return self._generate_legacy(prompt, max_tokens, temperature, **kwargs)
            else:
                logger.error("No available models for generation")
                raise RuntimeError("No available models for generation") from e
    
    def _generate_legacy(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt with enhanced error handling and input validation (legacy)."""
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
