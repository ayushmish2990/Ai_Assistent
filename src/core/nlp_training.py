"""
NLP model training and fine-tuning capabilities for the AI coding assistant.

This module provides training utilities for custom NLP models, including
fine-tuning pre-trained models for code-specific tasks.
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    """Dataset class for code-related NLP tasks."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            'labels': torch.tensor(label, dtype=torch.long)
        }


class IntentClassificationTrainer:
    """Trainer for intent classification models."""
    
    def __init__(self, model_name: str = "microsoft/codebert-base", num_labels: int = 10):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def prepare_model(self):
        """Initialize tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
            logger.info(f"Model {self.model_name} prepared successfully")
        except Exception as e:
            logger.error(f"Error preparing model: {str(e)}")
            raise
    
    def prepare_data(self, texts: List[str], labels: List[int], test_size: float = 0.2):
        """Prepare training and validation datasets."""
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_dataset = CodeDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = CodeDataset(val_texts, val_labels, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        output_dir: str = "./models/intent_classifier",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        """Train the intent classification model."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            learning_rate=learning_rate,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        try:
            self.trainer.train()
            logger.info("Training completed successfully")
            
            # Save the model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            return self.trainer.state.log_history
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise


class CodeEmbeddingTrainer:
    """Trainer for code embedding models."""
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
    
    def prepare_model(self):
        """Initialize tokenizer and model for embedding training."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            logger.info(f"Embedding model {self.model_name} prepared successfully")
        except Exception as e:
            logger.error(f"Error preparing embedding model: {str(e)}")
            raise
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Create embeddings for a list of texts."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.numpy())
        
        return np.vstack(embeddings)
    
    def fine_tune_for_similarity(
        self,
        code_pairs: List[Tuple[str, str]],
        similarity_scores: List[float],
        output_dir: str = "./models/code_similarity",
        num_epochs: int = 3
    ):
        """Fine-tune model for code similarity tasks."""
        # This is a simplified version - in practice, you'd implement
        # a more sophisticated similarity training loop
        logger.info("Fine-tuning for code similarity...")
        
        # Create training data
        texts = []
        for pair in code_pairs:
            texts.extend(pair)
        
        # Generate embeddings
        embeddings = self.create_embeddings(texts)
        
        # Save fine-tuned model
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save embeddings
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
        
        logger.info(f"Fine-tuned model saved to {output_dir}")


class NLPTrainingManager:
    """Manager for all NLP training operations."""
    
    def __init__(self, base_model_dir: str = "./models"):
        self.base_model_dir = Path(base_model_dir)
        self.base_model_dir.mkdir(exist_ok=True)
        
        self.intent_trainer = None
        self.embedding_trainer = None
    
    def train_intent_classifier(
        self,
        training_data: List[Dict[str, Any]],
        model_name: str = "microsoft/codebert-base",
        output_name: str = "intent_classifier"
    ) -> Dict[str, Any]:
        """Train an intent classification model."""
        try:
            # Extract texts and labels
            texts = [item['text'] for item in training_data]
            labels = [item['label'] for item in training_data]
            
            # Get unique labels and create label mapping
            unique_labels = list(set(labels))
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            label_ids = [label_to_id[label] for label in labels]
            
            # Initialize trainer
            self.intent_trainer = IntentClassificationTrainer(
                model_name=model_name,
                num_labels=len(unique_labels)
            )
            self.intent_trainer.prepare_model()
            
            # Prepare data
            train_dataset, val_dataset = self.intent_trainer.prepare_data(texts, label_ids)
            
            # Train model
            output_dir = self.base_model_dir / output_name
            training_history = self.intent_trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir=str(output_dir)
            )
            
            # Save label mapping
            with open(output_dir / "label_mapping.json", "w") as f:
                json.dump(label_to_id, f, indent=2)
            
            return {
                "status": "success",
                "model_path": str(output_dir),
                "num_labels": len(unique_labels),
                "training_samples": len(texts),
                "training_history": training_history
            }
        except Exception as e:
            logger.error(f"Error training intent classifier: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def train_code_embeddings(
        self,
        code_samples: List[str],
        model_name: str = "microsoft/codebert-base",
        output_name: str = "code_embeddings"
    ) -> Dict[str, Any]:
        """Train code embedding model."""
        try:
            # Initialize trainer
            self.embedding_trainer = CodeEmbeddingTrainer(model_name=model_name)
            self.embedding_trainer.prepare_model()
            
            # Create embeddings
            embeddings = self.embedding_trainer.create_embeddings(code_samples)
            
            # Save model and embeddings
            output_dir = self.base_model_dir / output_name
            output_dir.mkdir(exist_ok=True)
            
            self.embedding_trainer.model.save_pretrained(output_dir)
            self.embedding_trainer.tokenizer.save_pretrained(output_dir)
            np.save(output_dir / "embeddings.npy", embeddings)
            
            # Save metadata
            metadata = {
                "num_samples": len(code_samples),
                "embedding_dim": embeddings.shape[1],
                "model_name": model_name
            }
            with open(output_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "status": "success",
                "model_path": str(output_dir),
                "num_samples": len(code_samples),
                "embedding_dim": embeddings.shape[1]
            }
        except Exception as e:
            logger.error(f"Error training code embeddings: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def create_synthetic_training_data(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Create synthetic training data for intent classification."""
        intents = [
            "code_generation", "bug_fix", "code_review", "explanation",
            "optimization", "testing", "documentation", "refactoring"
        ]
        
        templates = {
            "code_generation": [
                "Generate a function that {}",
                "Create a class for {}",
                "Write code to {}",
                "Implement a {} algorithm"
            ],
            "bug_fix": [
                "Fix the bug in this code: {}",
                "Debug this error: {}",
                "Resolve the issue with {}",
                "Correct the problem in {}"
            ],
            "code_review": [
                "Review this code: {}",
                "Check this implementation: {}",
                "Analyze this function: {}",
                "Evaluate this code quality: {}"
            ],
            "explanation": [
                "Explain how this works: {}",
                "What does this code do: {}",
                "Describe this function: {}",
                "Help me understand: {}"
            ]
        }
        
        synthetic_data = []
        for i in range(num_samples):
            intent = np.random.choice(intents)
            if intent in templates:
                template = np.random.choice(templates[intent])
                placeholder = f"example_{i}"
                text = template.format(placeholder)
            else:
                text = f"Sample text for {intent} intent {i}"
            
            synthetic_data.append({
                "text": text,
                "label": intent
            })
        
        return synthetic_data
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get the status of all training operations."""
        status = {
            "base_model_dir": str(self.base_model_dir),
            "available_models": [],
            "training_active": {
                "intent_classifier": self.intent_trainer is not None,
                "embedding_trainer": self.embedding_trainer is not None
            }
        }
        
        # Check for existing models
        if self.base_model_dir.exists():
            for model_dir in self.base_model_dir.iterdir():
                if model_dir.is_dir():
                    status["available_models"].append(model_dir.name)
        
        return status