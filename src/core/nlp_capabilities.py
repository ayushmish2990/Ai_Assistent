"""NLP-specific AI capabilities for the coding assistant.

This module provides specialized NLP capabilities including intent classification,
code understanding, and semantic search functionality.
"""

from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
import logging

if TYPE_CHECKING:
    from .ai_capabilities import AICapability, CapabilityType, ProgrammingLanguage
    from .nlp_engine import NLPEngine
    from .nlp_models import NLPModelManager
    from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class NLPCapability:
    """Base NLP capability providing natural language processing features."""
    
    def __init__(self, model_manager, nlp_engine):
        """Initialize NLP capability with model manager and NLP engine."""
        self.model_manager = model_manager
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
    
    def get_capability_type(self):
        """Get the capability type."""
        # Import here to avoid circular import
        from .ai_capabilities import CapabilityType
        return CapabilityType.NATURAL_LANGUAGE_PROCESSING
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return ["python", "javascript", "java", "cpp", "c", "go", "rust", "typescript"]
    
    def execute(self, text: str, **kwargs) -> Dict[str, Any]:
        """Execute NLP processing on the given text."""
        try:
            # Basic NLP processing
            processed_text = self.nlp_engine.preprocess_text(text)
            tokens = self.nlp_engine.tokenize(processed_text)
            sentiment = self.nlp_engine.analyze_sentiment(processed_text)
            entities = self.nlp_engine.extract_entities(processed_text)
            
            return {
                "original_text": text,
                "processed_text": processed_text,
                "tokens": tokens,
                "sentiment": sentiment,
                "entities": entities,
                "token_count": len(tokens)
            }
        except Exception as e:
            self.logger.error(f"Error in NLP processing: {e}")
            return {"error": str(e)}


class IntentClassificationCapability:
    """Capability for classifying user intents in coding contexts."""
    
    def __init__(self, model_manager, nlp_model_manager):
        """Initialize intent classification capability."""
        self.model_manager = model_manager
        self.nlp_model_manager = nlp_model_manager
        self.logger = logging.getLogger(__name__)
    
    def get_capability_type(self):
        """Get the capability type."""
        # Import here to avoid circular import
        from .ai_capabilities import CapabilityType
        return CapabilityType.INTENT_CLASSIFICATION
    
    def execute(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """Classify the intent of user input."""
        try:
            # Get intent classifier
            intent_classifier = self.nlp_model_manager.get_intent_classifier()
            
            # Classify intent
            intent, confidence = intent_classifier.predict(user_input)
            
            # Map to coding-specific intents
            intent_mapping = {
                "generate": "CODE_GENERATION",
                "fix": "DEBUGGING", 
                "explain": "EXPLANATION",
                "optimize": "OPTIMIZATION",
                "refactor": "REFACTORING",
                "test": "TESTING"
            }
            
            # Check for keywords to improve classification
            user_lower = user_input.lower()
            if any(word in user_lower for word in ["create", "generate", "write", "implement"]):
                intent = "CODE_GENERATION"
                confidence = max(confidence, 0.8)
            elif any(word in user_lower for word in ["fix", "debug", "error", "bug"]):
                intent = "DEBUGGING"
                confidence = max(confidence, 0.8)
            elif any(word in user_lower for word in ["explain", "understand", "what does"]):
                intent = "EXPLANATION"
                confidence = max(confidence, 0.8)
            elif any(word in user_lower for word in ["optimize", "improve", "faster", "performance"]):
                intent = "OPTIMIZATION"
                confidence = max(confidence, 0.8)
            elif any(word in user_lower for word in ["refactor", "restructure", "clean"]):
                intent = "REFACTORING"
                confidence = max(confidence, 0.8)
            
            return {
                "user_input": user_input,
                "intent": intent_mapping.get(intent.lower(), intent),
                "confidence": confidence,
                "suggested_actions": self._get_suggested_actions(intent)
            }
        except Exception as e:
            self.logger.error(f"Error in intent classification: {e}")
            return {
                "user_input": user_input,
                "intent": "UNKNOWN",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _get_suggested_actions(self, intent: str) -> List[str]:
        """Get suggested actions based on classified intent."""
        actions_map = {
            "CODE_GENERATION": [
                "Analyze requirements",
                "Generate code structure", 
                "Add documentation",
                "Include error handling"
            ],
            "DEBUGGING": [
                "Analyze error messages",
                "Check code logic",
                "Suggest fixes",
                "Add debugging statements"
            ],
            "EXPLANATION": [
                "Break down code structure",
                "Explain algorithms",
                "Provide examples",
                "Highlight key concepts"
            ],
            "OPTIMIZATION": [
                "Analyze performance bottlenecks",
                "Suggest algorithmic improvements",
                "Recommend data structure changes",
                "Profile code execution"
            ],
            "REFACTORING": [
                "Identify code smells",
                "Suggest design patterns",
                "Improve code organization",
                "Enhance readability"
            ]
        }
        return actions_map.get(intent, ["Provide general assistance"])


class CodeUnderstandingCapability:
    """Capability for understanding and analyzing code structure and semantics."""
    
    def __init__(self, model_manager, nlp_engine):
        """Initialize code understanding capability."""
        self.model_manager = model_manager
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
    
    def get_capability_type(self):
        """Get the capability type."""
        # Import here to avoid circular import
        from .ai_capabilities import CapabilityType
        return CapabilityType.CODE_UNDERSTANDING
    
    def execute(self, code: str, language: str = "python", **kwargs) -> Dict[str, Any]:
        """Analyze and understand the given code."""
        try:
            # Extract code context
            context = self.nlp_engine.extract_code_context(code, language)
            
            # Analyze code structure
            structure = self._analyze_code_structure(code, language)
            
            # Extract semantic information
            semantic_info = self.nlp_engine.extract_semantic_info(code, language)
            
            # Analyze complexity
            complexity = self.nlp_engine.analyze_code_complexity(code, language)
            
            return {
                "code": code,
                "language": language,
                "context": context,
                "structure": structure,
                "semantic_info": semantic_info,
                "complexity": complexity,
                "functions": structure.get("functions", []),
                "classes": structure.get("classes", []),
                "imports": structure.get("imports", []),
                "summary": self._generate_code_summary(structure, semantic_info),
                "purpose": semantic_info.get("purpose", "Code analysis")
            }
        except Exception as e:
            self.logger.error(f"Error in code understanding: {e}")
            return {
                "code": code,
                "language": language,
                "error": str(e)
            }
    
    def _analyze_code_structure(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze the structural elements of code."""
        structure = {
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [],
            "comments": []
        }
        
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Detect functions
            if language == "python":
                if line.startswith("def "):
                    func_name = line.split("(")[0].replace("def ", "")
                    structure["functions"].append(func_name)
                elif line.startswith("class "):
                    class_name = line.split(":")[0].replace("class ", "")
                    structure["classes"].append(class_name)
                elif line.startswith("import ") or line.startswith("from "):
                    structure["imports"].append(line)
            elif language in ["javascript", "typescript"]:
                if "function " in line or "=>" in line:
                    structure["functions"].append("function")
                elif "class " in line:
                    structure["classes"].append("class")
                elif "import " in line or "require(" in line:
                    structure["imports"].append(line)
            
            # Detect comments
            if line.startswith("#") or line.startswith("//") or line.startswith("/*"):
                structure["comments"].append(line)
        
        return structure
    
    def _generate_code_summary(self, structure: Dict[str, Any], semantic_info: Dict[str, Any]) -> str:
        """Generate a summary of the code."""
        summary_parts = []
        
        if structure["functions"]:
            summary_parts.append(f"Contains {len(structure['functions'])} function(s)")
        
        if structure["classes"]:
            summary_parts.append(f"Contains {len(structure['classes'])} class(es)")
        
        if structure["imports"]:
            summary_parts.append(f"Uses {len(structure['imports'])} import(s)")
        
        purpose = semantic_info.get("purpose", "")
        if purpose:
            summary_parts.append(f"Purpose: {purpose}")
        
        return ". ".join(summary_parts) if summary_parts else "Code structure analyzed"


class SemanticSearchCapability:
    """Capability for semantic search in code and documentation."""
    
    def __init__(self, model_manager, nlp_engine):
        """Initialize semantic search capability."""
        self.model_manager = model_manager
        self.nlp_engine = nlp_engine
        self.logger = logging.getLogger(__name__)
    
    def get_capability_type(self):
        """Get the capability type."""
        # Import here to avoid circular import
        from .ai_capabilities import CapabilityType
        return CapabilityType.SEMANTIC_SEARCH
    
    def execute(self, query: str, documents: List[str], top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Perform semantic search on the given documents."""
        try:
            # Get embeddings for query and documents
            query_embedding = self.nlp_engine.get_embeddings([query])[0]
            doc_embeddings = self.nlp_engine.get_embeddings(documents)
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                similarity = self.nlp_engine.calculate_similarity(query_embedding, doc_embedding)
                similarities.append({
                    "index": i,
                    "document": documents[i],
                    "score": similarity
                })
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top-k results
            top_results = similarities[:top_k]
            
            return {
                "query": query,
                "total_documents": len(documents),
                "results": top_results,
                "search_metadata": {
                    "top_k": top_k,
                    "max_score": top_results[0]["score"] if top_results else 0.0,
                    "min_score": top_results[-1]["score"] if top_results else 0.0
                }
            }
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return {
                "query": query,
                "results": [],
                "error": str(e)
            }