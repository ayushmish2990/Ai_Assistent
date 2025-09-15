"""Enhanced Model Manager supporting multiple AI providers."""
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .config import Config

logger = logging.getLogger(__name__)

class BaseModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        pass

class OpenAIProvider(BaseModelProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if OPENAI_AVAILABLE:
            self.client = openai.OpenAI(
                api_key=config.get('api_key') or os.getenv('OPENAI_API_KEY')
            )
        else:
            self.client = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.get('model_name', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', self.config.get('max_tokens', 1000)),
                temperature=kwargs.get('temperature', self.config.get('temperature', 0.7)),
                top_p=kwargs.get('top_p', self.config.get('top_p', 0.9))
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return OPENAI_AVAILABLE and (self.config.get('api_key') or os.getenv('OPENAI_API_KEY'))

class HuggingFaceProvider(BaseModelProvider):
    """Hugging Face API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('HUGGINGFACE_API_KEY')
        self.api_url = f"https://api-inference.huggingface.co/models/{config.get('model_name', 'microsoft/DialoGPT-medium')}"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Hugging Face Inference API."""
        if not self.api_key:
            raise RuntimeError("Hugging Face API key not available")
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get('max_tokens', self.config.get('max_tokens', 100)),
                "temperature": kwargs.get('temperature', self.config.get('temperature', 0.7)),
                "top_p": kwargs.get('top_p', self.config.get('top_p', 0.9)),
                "do_sample": True
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Remove the input prompt from the response
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text
            else:
                return "No response generated"
                
        except Exception as e:
            logger.error(f"Hugging Face API error: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Hugging Face is available."""
        return bool(self.api_key)

class AnthropicProvider(BaseModelProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(
                api_key=config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
            )
        else:
            self.client = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic Claude API."""
        if not self.client:
            raise RuntimeError("Anthropic client not available")
        
        try:
            response = self.client.messages.create(
                model=self.config.get('model_name', 'claude-3-sonnet-20240229'),
                max_tokens=kwargs.get('max_tokens', self.config.get('max_tokens', 1000)),
                temperature=kwargs.get('temperature', self.config.get('temperature', 0.7)),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return ANTHROPIC_AVAILABLE and (self.config.get('api_key') or os.getenv('ANTHROPIC_API_KEY'))

class LocalModelProvider(BaseModelProvider):
    """Local model provider using transformers."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = None
        
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._load_model()
    
    def _load_model(self) -> None:
        """Load local model and tokenizer."""
        try:
            model_name = self.config.get('model_name', 'microsoft/DialoGPT-small')
            logger.info(f"Loading local model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left",
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"Local model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using local model."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Local model not loaded")
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length or 1024,
            ).to(self.device)
            
            generation_config = GenerationConfig(
                max_new_tokens=kwargs.get('max_tokens', self.config.get('max_tokens', 100)),
                temperature=kwargs.get('temperature', self.config.get('temperature', 0.7)),
                top_p=kwargs.get('top_p', self.config.get('top_p', 0.9)),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                )
            
            generated = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )
            
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            
            return generated
            
        except Exception as e:
            logger.error(f"Local model generation error: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if local model is available."""
        return TRANSFORMERS_AVAILABLE and self.model is not None

class MultiProviderModelManager:
    """Enhanced model manager supporting multiple AI providers."""
    
    def __init__(self, config: Config):
        self.config = config
        self.providers = {}
        self.current_provider = None
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize all available providers."""
        providers_config = getattr(self.config, 'providers', None)
        
        # Initialize OpenAI provider
        if (providers_config and hasattr(providers_config, 'openai') and providers_config.openai.api_key) or os.getenv('OPENAI_API_KEY'):
            openai_config = providers_config.openai.dict() if providers_config and hasattr(providers_config, 'openai') else {}
            if os.getenv('OPENAI_API_KEY'):
                openai_config['api_key'] = os.getenv('OPENAI_API_KEY')
            self.providers['openai'] = OpenAIProvider(openai_config)
        
        # Initialize Hugging Face provider
        if (providers_config and hasattr(providers_config, 'huggingface') and providers_config.huggingface.api_key) or os.getenv('HUGGINGFACE_API_KEY'):
            hf_config = providers_config.huggingface.dict() if providers_config and hasattr(providers_config, 'huggingface') else {}
            if os.getenv('HUGGINGFACE_API_KEY'):
                hf_config['api_key'] = os.getenv('HUGGINGFACE_API_KEY')
            self.providers['huggingface'] = HuggingFaceProvider(hf_config)
        
        # Initialize Anthropic provider
        if (providers_config and hasattr(providers_config, 'anthropic') and providers_config.anthropic.api_key) or os.getenv('ANTHROPIC_API_KEY'):
            anthropic_config = providers_config.anthropic.dict() if providers_config and hasattr(providers_config, 'anthropic') else {}
            if os.getenv('ANTHROPIC_API_KEY'):
                anthropic_config['api_key'] = os.getenv('ANTHROPIC_API_KEY')
            self.providers['anthropic'] = AnthropicProvider(anthropic_config)
        
        # Initialize local model provider
        if TRANSFORMERS_AVAILABLE:
            local_config = providers_config.local.dict() if providers_config and hasattr(providers_config, 'local') else {'model_name': 'microsoft/DialoGPT-small'}
            self.providers['local'] = LocalModelProvider(local_config)
        
        # Set default provider
        self._set_default_provider()
    
    def _set_default_provider(self) -> None:
        """Set the default provider based on availability."""
        # Priority order: OpenAI, Anthropic, Hugging Face, Local
        priority_order = ['openai', 'anthropic', 'huggingface', 'local']
        
        for provider_name in priority_order:
            if provider_name in self.providers and self.providers[provider_name].is_available():
                self.current_provider = provider_name
                logger.info(f"Using {provider_name} as default provider")
                return
        
        logger.warning("No available providers found")
    
    def set_provider(self, provider_name: str) -> bool:
        """Set the current provider."""
        if provider_name in self.providers and self.providers[provider_name].is_available():
            self.current_provider = provider_name
            logger.info(f"Switched to provider: {provider_name}")
            return True
        logger.warning(f"Provider {provider_name} not available")
        return False
    
    def set_default_provider(self, provider_name: str) -> bool:
        """Set the default provider (public method)."""
        return self.set_provider(provider_name)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return [name for name, provider in self.providers.items() if provider.is_available()]
    
    def generate(self, prompt: str, provider: Optional[str] = None, **kwargs) -> str:
        """Generate text using the specified or current provider."""
        provider_name = provider or self.current_provider
        
        if not provider_name or provider_name not in self.providers:
            raise RuntimeError(f"Provider {provider_name} not available")
        
        provider_instance = self.providers[provider_name]
        if not provider_instance.is_available():
            raise RuntimeError(f"Provider {provider_name} is not properly configured")
        
        logger.info(f"Generating response using {provider_name} provider")
        return provider_instance.generate(prompt, **kwargs)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about all providers."""
        info = {
            'current_provider': self.current_provider,
            'available_providers': self.get_available_providers(),
            'provider_status': {}
        }
        
        for name, provider in self.providers.items():
            info['provider_status'][name] = {
                'available': provider.is_available(),
                'config': provider.config
            }
        
        return info