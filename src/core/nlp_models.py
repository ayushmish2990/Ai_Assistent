"""Specialized NLP Models for Code Understanding and Generation.

This module contains custom neural network models and fine-tuned transformers
specifically designed for code-related natural language processing tasks.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        AutoModelForCausalLM, TrainingArguments, Trainer,
        BertModel, BertTokenizer, RobertaModel, RobertaTokenizer,
        GPT2Model, GPT2Tokenizer, T5Model, T5Tokenizer
    )
    from sentence_transformers import SentenceTransformer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .nlp_engine import IntentType, LanguageType

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for NLP models."""
    model_name: str
    max_length: int = 512
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 6
    dropout: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01

class CodeIntentClassifier(nn.Module):
    """Neural network for classifying coding-related intents."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_labels = len(IntentType)
        
        # Base transformer model
        self.transformer = AutoModel.from_pretrained(config.model_name)
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # Intent type mapping
        self.intent_to_id = {intent: i for i, intent in enumerate(IntentType)}
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state
        }
    
    def predict(self, text: str, tokenizer) -> Tuple[IntentType, float]:
        """Predict intent for given text."""
        self.eval()
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.forward(**inputs)
            probabilities = F.softmax(outputs['logits'], dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
        
        predicted_intent = self.id_to_intent[predicted_id]
        return predicted_intent, confidence

class CodeLanguageDetector(nn.Module):
    """Neural network for detecting programming languages from text."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_labels = len(LanguageType)
        
        # Base transformer model
        self.transformer = AutoModel.from_pretrained(config.model_name)
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # Language type mapping
        self.lang_to_id = {lang: i for i, lang in enumerate(LanguageType)}
        self.id_to_lang = {i: lang for lang, i in self.lang_to_id.items()}
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state
        }
    
    def predict(self, text: str, tokenizer) -> Tuple[LanguageType, float]:
        """Predict programming language for given text."""
        self.eval()
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.forward(**inputs)
            probabilities = F.softmax(outputs['logits'], dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
        
        predicted_language = self.id_to_lang[predicted_id]
        return predicted_language, confidence

class CodeSemanticEmbedder(nn.Module):
    """Model for generating semantic embeddings for code and text."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Base transformer model
        self.transformer = AutoModel.from_pretrained(config.model_name)
        
        # Projection layers for different modalities
        self.text_projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.code_projector = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Contrastive learning head
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def encode_text(self, input_ids, attention_mask=None):
        """Encode text into semantic embeddings."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Mean pooling
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        embeddings = self.text_projector(embeddings)
        
        return F.normalize(embeddings, p=2, dim=1)
    
    def encode_code(self, input_ids, attention_mask=None):
        """Encode code into semantic embeddings."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Mean pooling
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        embeddings = self.code_projector(embeddings)
        
        return F.normalize(embeddings, p=2, dim=1)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to token embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, text_input_ids, text_attention_mask, code_input_ids, code_attention_mask):
        """Forward pass for contrastive learning."""
        text_embeddings = self.encode_text(text_input_ids, text_attention_mask)
        code_embeddings = self.encode_code(code_input_ids, code_attention_mask)
        
        # Compute similarity matrix
        logits = torch.matmul(text_embeddings, code_embeddings.t()) * torch.exp(self.temperature)
        
        # Labels for contrastive learning (diagonal should be positive pairs)
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Compute contrastive loss
        loss_text_to_code = F.cross_entropy(logits, labels)
        loss_code_to_text = F.cross_entropy(logits.t(), labels)
        loss = (loss_text_to_code + loss_code_to_text) / 2
        
        return {
            'loss': loss,
            'text_embeddings': text_embeddings,
            'code_embeddings': code_embeddings,
            'logits': logits
        }

class CodeComplexityEstimator(nn.Module):
    """Model for estimating code complexity from natural language descriptions."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Base transformer model
        self.transformer = AutoModel.from_pretrained(config.model_name)
        
        # Regression head for complexity estimation
        self.dropout = nn.Dropout(config.dropout)
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        complexity_score = self.regressor(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(complexity_score.squeeze(), labels.float())
        
        return {
            'loss': loss,
            'complexity_score': complexity_score,
            'hidden_states': outputs.last_hidden_state
        }
    
    def predict(self, text: str, tokenizer) -> float:
        """Predict complexity score for given text."""
        self.eval()
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.forward(**inputs)
            complexity_score = torch.sigmoid(outputs['complexity_score']).item()
        
        return complexity_score

class CodeNERModel(nn.Module):
    """Named Entity Recognition model for code-related entities."""
    
    def __init__(self, config: ModelConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # Base transformer model
        self.transformer = AutoModel.from_pretrained(config.model_name)
        
        # Token classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state
        }

class NLPModelManager:
    """Manager for all NLP models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the model manager."""
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        
        # Model configurations
        self.model_configs = {
            'intent_classifier': ModelConfig(
                model_name=self.config.get('intent_model', 'bert-base-uncased'),
                max_length=256,
                hidden_size=768
            ),
            'language_detector': ModelConfig(
                model_name=self.config.get('language_model', 'bert-base-uncased'),
                max_length=512,
                hidden_size=768
            ),
            'semantic_embedder': ModelConfig(
                model_name=self.config.get('embedding_model', 'bert-base-uncased'),
                max_length=512,
                hidden_size=768
            ),
            'complexity_estimator': ModelConfig(
                model_name=self.config.get('complexity_model', 'bert-base-uncased'),
                max_length=256,
                hidden_size=768
            )
        }
        
        # Initialize models
        self.models = {}
        self.tokenizers = {}
        
        if TORCH_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models."""
        try:
            # Intent classifier
            intent_config = self.model_configs['intent_classifier']
            self.models['intent_classifier'] = CodeIntentClassifier(intent_config).to(self.device)
            self.tokenizers['intent_classifier'] = AutoTokenizer.from_pretrained(intent_config.model_name)
            
            # Language detector
            lang_config = self.model_configs['language_detector']
            self.models['language_detector'] = CodeLanguageDetector(lang_config).to(self.device)
            self.tokenizers['language_detector'] = AutoTokenizer.from_pretrained(lang_config.model_name)
            
            # Semantic embedder
            embed_config = self.model_configs['semantic_embedder']
            self.models['semantic_embedder'] = CodeSemanticEmbedder(embed_config).to(self.device)
            self.tokenizers['semantic_embedder'] = AutoTokenizer.from_pretrained(embed_config.model_name)
            
            # Complexity estimator
            complex_config = self.model_configs['complexity_estimator']
            self.models['complexity_estimator'] = CodeComplexityEstimator(complex_config).to(self.device)
            self.tokenizers['complexity_estimator'] = AutoTokenizer.from_pretrained(complex_config.model_name)
            
            logger.info("All NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def predict_intent(self, text: str) -> Tuple[IntentType, float]:
        """Predict intent using the intent classifier."""
        if 'intent_classifier' in self.models:
            model = self.models['intent_classifier']
            tokenizer = self.tokenizers['intent_classifier']
            return model.predict(text, tokenizer)
        
        return IntentType.UNKNOWN, 0.0
    
    def detect_language(self, text: str) -> Tuple[LanguageType, float]:
        """Detect programming language using the language detector."""
        if 'language_detector' in self.models:
            model = self.models['language_detector']
            tokenizer = self.tokenizers['language_detector']
            return model.predict(text, tokenizer)
        
        return LanguageType.UNKNOWN, 0.0
    
    def get_semantic_embeddings(self, text: str, is_code: bool = False) -> Optional[np.ndarray]:
        """Get semantic embeddings for text or code."""
        if 'semantic_embedder' not in self.models:
            return None
        
        model = self.models['semantic_embedder']
        tokenizer = self.tokenizers['semantic_embedder']
        
        model.eval()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            if is_code:
                embeddings = model.encode_code(**inputs)
            else:
                embeddings = model.encode_text(**inputs)
        
        return embeddings.cpu().numpy()[0]
    
    def estimate_complexity(self, text: str) -> float:
        """Estimate complexity score for text."""
        if 'complexity_estimator' in self.models:
            model = self.models['complexity_estimator']
            tokenizer = self.tokenizers['complexity_estimator']
            return model.predict(text, tokenizer)
        
        return 0.5  # Default medium complexity
    
    def save_models(self, save_dir: str):
        """Save all models to directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = save_path / f"{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Save tokenizer
            tokenizer_path = save_path / f"{model_name}_tokenizer"
            self.tokenizers[model_name].save_pretrained(tokenizer_path)
        
        # Save configurations
        config_path = save_path / "model_configs.json"
        with open(config_path, 'w') as f:
            json.dump({name: config.__dict__ for name, config in self.model_configs.items()}, f, indent=2)
        
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir: str):
        """Load models from directory."""
        load_path = Path(load_dir)
        
        if not load_path.exists():
            logger.error(f"Model directory {load_dir} does not exist")
            return
        
        # Load configurations
        config_path = load_path / "model_configs.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                configs = json.load(f)
                for name, config_dict in configs.items():
                    self.model_configs[name] = ModelConfig(**config_dict)
        
        # Load models
        for model_name in self.models.keys():
            model_path = load_path / f"{model_name}.pt"
            if model_path.exists():
                self.models[model_name].load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded {model_name}")
            
            # Load tokenizer
            tokenizer_path = load_path / f"{model_name}_tokenizer"
            if tokenizer_path.exists():
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(tokenizer_path)
        
        logger.info(f"Models loaded from {load_dir}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'torch_available': TORCH_AVAILABLE,
            'device': str(self.device) if self.device else 'cpu',
            'models_loaded': list(self.models.keys()),
            'model_configs': {name: config.__dict__ for name, config in self.model_configs.items()}
        }

# Training utilities
class CodeIntentDataset(Dataset):
    """Dataset for training intent classification model."""
    
    def __init__(self, texts: List[str], labels: List[IntentType], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        self.intent_to_id = {intent: i for i, intent in enumerate(IntentType)}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.intent_to_id[label], dtype=torch.long)
        }

def train_intent_classifier(
    model: CodeIntentClassifier,
    train_dataset: CodeIntentDataset,
    val_dataset: CodeIntentDataset,
    config: ModelConfig
) -> CodeIntentClassifier:
    """Train the intent classification model."""
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    return model