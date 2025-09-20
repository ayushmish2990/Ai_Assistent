"""Advanced NLP Engine for AI Coding Assistant.

This module provides comprehensive Natural Language Processing capabilities including:
- Text preprocessing and tokenization
- Semantic embeddings and similarity
- Intent recognition and classification
- Named Entity Recognition (NER)
- Sentiment analysis
- Language understanding and context extraction
- Code-text alignment and understanding
"""

import logging
import re
import string
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        pipeline, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
    )
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Types of user intents for code assistance."""
    CODE_GENERATION = "code_generation"
    CODE_EXPLANATION = "code_explanation"
    CODE_DEBUGGING = "code_debugging"
    CODE_OPTIMIZATION = "code_optimization"
    CODE_REFACTORING = "code_refactoring"
    CODE_TESTING = "code_testing"
    DOCUMENTATION = "documentation"
    QUESTION_ANSWERING = "question_answering"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"

class LanguageType(Enum):
    """Programming languages for context understanding."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    UNKNOWN = "unknown"

@dataclass
class NLPResult:
    """Result from NLP processing."""
    original_text: str
    processed_text: str
    tokens: List[str]
    embeddings: Optional[np.ndarray] = None
    intent: IntentType = IntentType.UNKNOWN
    confidence: float = 0.0
    entities: List[Dict[str, Any]] = None
    sentiment: Dict[str, float] = None
    language_context: Optional[LanguageType] = None
    keywords: List[str] = None
    semantic_features: Dict[str, Any] = None

@dataclass
class CodeContext:
    """Context information extracted from code-related text."""
    programming_language: LanguageType
    code_snippets: List[str]
    function_names: List[str]
    variable_names: List[str]
    libraries: List[str]
    concepts: List[str]
    complexity_level: str  # "beginner", "intermediate", "advanced"

class NLPEngine:
    """Advanced NLP engine for natural language understanding."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the NLP engine with models and processors."""
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TRANSFORMERS_AVAILABLE else None
        
        # Initialize models
        self.tokenizer = None
        self.embedding_model = None
        self.intent_classifier = None
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        self.sentence_transformer = None
        
        # Initialize NLTK components
        self.lemmatizer = None
        self.stop_words = set()
        
        # Initialize spaCy
        self.nlp_spacy = None
        
        # Load models
        self._initialize_models()
        self._initialize_nltk()
        self._initialize_spacy()
        
        # Code-related patterns
        self._setup_code_patterns()
        
        logger.info("NLP Engine initialized successfully")
    
    def _initialize_models(self):
        """Initialize transformer models."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Some NLP features will be disabled.")
            return
        
        try:
            # Sentence transformer for embeddings
            model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
            self.sentence_transformer = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer: {model_name}")
            
            # BERT-based model for general NLP tasks
            bert_model = self.config.get('bert_model', 'bert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.embedding_model = AutoModel.from_pretrained(bert_model).to(self.device)
            logger.info(f"Loaded BERT model: {bert_model}")
            
            # Intent classification pipeline
            self.intent_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
        except Exception as e:
            logger.error(f"Error initializing transformer models: {e}")
    
    def _initialize_nltk(self):
        """Initialize NLTK components."""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available. Some NLP features will be disabled.")
            return
        
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing NLTK: {e}")
    
    def _initialize_spacy(self):
        """Initialize spaCy model."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Some NLP features will be disabled.")
            return
        
        try:
            # Try to load English model
            self.nlp_spacy = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    
    def _setup_code_patterns(self):
        """Setup regex patterns for code-related text analysis."""
        self.code_patterns = {
            'function_call': re.compile(r'\b\w+\s*\([^)]*\)'),
            'variable_assignment': re.compile(r'\b\w+\s*=\s*[^=]'),
            'import_statement': re.compile(r'\b(?:import|from|include|require)\s+[\w.]+'),
            'class_definition': re.compile(r'\b(?:class|interface|struct)\s+\w+'),
            'function_definition': re.compile(r'\b(?:def|function|func|fn)\s+\w+'),
            'code_block': re.compile(r'```[\s\S]*?```|`[^`]+`'),
            'file_path': re.compile(r'[./]?[\w/\\.-]+\.\w+'),
            'url': re.compile(r'https?://[^\s]+'),
            'error_message': re.compile(r'\b(?:error|exception|traceback|failed|invalid)\b', re.IGNORECASE),
        }
        
        self.language_keywords = {
            LanguageType.PYTHON: ['def', 'class', 'import', 'from', 'if', 'else', 'for', 'while', 'try', 'except'],
            LanguageType.JAVASCRIPT: ['function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'try', 'catch'],
            LanguageType.JAVA: ['public', 'private', 'class', 'interface', 'if', 'else', 'for', 'while', 'try', 'catch'],
            LanguageType.CPP: ['int', 'char', 'class', 'struct', 'if', 'else', 'for', 'while', 'try', 'catch'],
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP analysis."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle code blocks separately
        code_blocks = self.code_patterns['code_block'].findall(text)
        text_without_code = self.code_patterns['code_block'].sub(' [CODE_BLOCK] ', text)
        
        # Basic cleaning while preserving important punctuation
        text_without_code = text_without_code.lower()
        
        return text_without_code
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using multiple approaches."""
        tokens = []
        
        if NLTK_AVAILABLE:
            # NLTK tokenization
            nltk_tokens = word_tokenize(text)
            tokens.extend(nltk_tokens)
        
        if self.tokenizer:
            # Transformer tokenization
            transformer_tokens = self.tokenizer.tokenize(text)
            tokens.extend(transformer_tokens)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        
        return unique_tokens
    
    def get_embeddings(self, text: str) -> Optional[np.ndarray]:
        """Generate embeddings for text."""
        if self.sentence_transformer:
            try:
                embeddings = self.sentence_transformer.encode([text])
                return embeddings[0]
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
        
        return None
    
    def classify_intent(self, text: str) -> Tuple[IntentType, float]:
        """Classify user intent from text."""
        # Rule-based intent classification
        text_lower = text.lower()
        
        # Code generation patterns
        if any(phrase in text_lower for phrase in ['write', 'create', 'generate', 'make', 'build']):
            if any(word in text_lower for word in ['function', 'class', 'code', 'script', 'program']):
                return IntentType.CODE_GENERATION, 0.8
        
        # Code explanation patterns
        if any(phrase in text_lower for phrase in ['explain', 'what does', 'how does', 'understand']):
            return IntentType.CODE_EXPLANATION, 0.8
        
        # Debugging patterns
        if any(phrase in text_lower for phrase in ['debug', 'fix', 'error', 'bug', 'issue', 'problem']):
            return IntentType.CODE_DEBUGGING, 0.8
        
        # Optimization patterns
        if any(phrase in text_lower for phrase in ['optimize', 'improve', 'faster', 'efficient', 'performance']):
            return IntentType.CODE_OPTIMIZATION, 0.7
        
        # Testing patterns
        if any(phrase in text_lower for phrase in ['test', 'unit test', 'testing', 'assert']):
            return IntentType.CODE_TESTING, 0.8
        
        # Documentation patterns
        if any(phrase in text_lower for phrase in ['document', 'comment', 'docstring', 'readme']):
            return IntentType.DOCUMENTATION, 0.7
        
        # Question patterns
        if text.strip().endswith('?') or text_lower.startswith(('what', 'how', 'why', 'when', 'where')):
            return IntentType.QUESTION_ANSWERING, 0.6
        
        return IntentType.GENERAL_CHAT, 0.3
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        entities = []
        
        # Use transformer-based NER if available
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text)
                for entity in ner_results:
                    entities.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': entity['score'],
                        'start': entity.get('start', 0),
                        'end': entity.get('end', 0)
                    })
            except Exception as e:
                logger.error(f"Error in NER pipeline: {e}")
        
        # Extract code-specific entities
        code_entities = self._extract_code_entities(text)
        entities.extend(code_entities)
        
        return entities
    
    def _extract_code_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract code-specific entities."""
        entities = []
        
        # Function calls
        for match in self.code_patterns['function_call'].finditer(text):
            entities.append({
                'text': match.group(),
                'label': 'FUNCTION_CALL',
                'confidence': 0.9,
                'start': match.start(),
                'end': match.end()
            })
        
        # Import statements
        for match in self.code_patterns['import_statement'].finditer(text):
            entities.append({
                'text': match.group(),
                'label': 'IMPORT',
                'confidence': 0.9,
                'start': match.start(),
                'end': match.end()
            })
        
        # File paths
        for match in self.code_patterns['file_path'].finditer(text):
            entities.append({
                'text': match.group(),
                'label': 'FILE_PATH',
                'confidence': 0.8,
                'start': match.start(),
                'end': match.end()
            })
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text)[0]
                return {
                    'label': result['label'],
                    'score': result['score']
                }
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
        
        return {'label': 'NEUTRAL', 'score': 0.5}
    
    def detect_programming_language(self, text: str) -> LanguageType:
        """Detect programming language from text context."""
        text_lower = text.lower()
        
        # Count language-specific keywords
        language_scores = {}
        for lang, keywords in self.language_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                language_scores[lang] = score
        
        if language_scores:
            return max(language_scores, key=language_scores.get)
        
        # Check for explicit language mentions
        if 'python' in text_lower:
            return LanguageType.PYTHON
        elif 'javascript' in text_lower or 'js' in text_lower:
            return LanguageType.JAVASCRIPT
        elif 'java' in text_lower:
            return LanguageType.JAVA
        elif 'c++' in text_lower or 'cpp' in text_lower:
            return LanguageType.CPP
        
        return LanguageType.UNKNOWN
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        keywords = []
        
        if NLTK_AVAILABLE and self.lemmatizer:
            # Tokenize and remove stop words
            tokens = word_tokenize(text.lower())
            filtered_tokens = [token for token in tokens if token not in self.stop_words and token.isalpha()]
            
            # Lemmatize
            lemmatized = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
            
            # Get POS tags and keep nouns and verbs
            pos_tags = pos_tag(lemmatized)
            keywords = [word for word, pos in pos_tags if pos.startswith(('NN', 'VB'))]
        
        # Add code-specific keywords
        code_keywords = []
        for pattern_name, pattern in self.code_patterns.items():
            matches = pattern.findall(text)
            code_keywords.extend(matches)
        
        keywords.extend(code_keywords)
        
        return list(set(keywords))  # Remove duplicates
    
    def extract_code_context(self, text: str) -> CodeContext:
        """Extract code-specific context from text."""
        programming_language = self.detect_programming_language(text)
        
        # Extract code snippets
        code_snippets = self.code_patterns['code_block'].findall(text)
        
        # Extract function names
        function_names = []
        func_matches = self.code_patterns['function_definition'].findall(text)
        for match in func_matches:
            # Extract function name from definition
            name_match = re.search(r'\b(?:def|function|func|fn)\s+(\w+)', match)
            if name_match:
                function_names.append(name_match.group(1))
        
        # Extract variable names (simplified)
        variable_names = []
        var_matches = self.code_patterns['variable_assignment'].findall(text)
        for match in var_matches:
            var_name = match.split('=')[0].strip()
            if var_name.isidentifier():
                variable_names.append(var_name)
        
        # Extract libraries/imports
        libraries = []
        import_matches = self.code_patterns['import_statement'].findall(text)
        for match in import_matches:
            # Extract library name
            lib_match = re.search(r'(?:import|from|require)\s+([\w.]+)', match)
            if lib_match:
                libraries.append(lib_match.group(1))
        
        # Determine complexity level
        complexity_indicators = {
            'beginner': ['hello', 'world', 'basic', 'simple', 'easy', 'tutorial'],
            'intermediate': ['class', 'function', 'loop', 'condition', 'array', 'object'],
            'advanced': ['algorithm', 'optimization', 'async', 'threading', 'database', 'api']
        }
        
        text_lower = text.lower()
        complexity_scores = {}
        for level, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            complexity_scores[level] = score
        
        complexity_level = max(complexity_scores, key=complexity_scores.get) if complexity_scores else 'intermediate'
        
        # Extract programming concepts
        concepts = []
        concept_patterns = {
            'oop': r'\b(?:class|object|inheritance|polymorphism|encapsulation)\b',
            'functional': r'\b(?:lambda|map|filter|reduce|closure)\b',
            'async': r'\b(?:async|await|promise|callback)\b',
            'data_structures': r'\b(?:array|list|dict|set|tree|graph|stack|queue)\b',
            'algorithms': r'\b(?:sort|search|recursion|dynamic programming|greedy)\b'
        }
        
        for concept, pattern in concept_patterns.items():
            if re.search(pattern, text_lower):
                concepts.append(concept)
        
        return CodeContext(
            programming_language=programming_language,
            code_snippets=code_snippets,
            function_names=function_names,
            variable_names=variable_names,
            libraries=libraries,
            concepts=concepts,
            complexity_level=complexity_level
        )
    
    def process(self, text: str) -> NLPResult:
        """Process text through the complete NLP pipeline."""
        if not text:
            return NLPResult(
                original_text="",
                processed_text="",
                tokens=[],
                intent=IntentType.UNKNOWN,
                confidence=0.0
            )
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        tokens = self.tokenize(processed_text)
        
        # Generate embeddings
        embeddings = self.get_embeddings(text)
        
        # Classify intent
        intent, confidence = self.classify_intent(text)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Detect programming language context
        language_context = self.detect_programming_language(text)
        
        # Extract keywords
        keywords = self.extract_keywords(text)
        
        # Extract code context
        code_context = self.extract_code_context(text)
        
        # Create semantic features
        semantic_features = {
            'code_context': code_context.__dict__,
            'has_code': bool(code_context.code_snippets),
            'complexity': code_context.complexity_level,
            'concepts': code_context.concepts,
            'entity_count': len(entities),
            'keyword_count': len(keywords)
        }
        
        return NLPResult(
            original_text=text,
            processed_text=processed_text,
            tokens=tokens,
            embeddings=embeddings,
            intent=intent,
            confidence=confidence,
            entities=entities,
            sentiment=sentiment,
            language_context=language_context,
            keywords=keywords,
            semantic_features=semantic_features
        )
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if self.sentence_transformer:
            try:
                embeddings = self.sentence_transformer.encode([text1, text2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return float(similarity)
            except Exception as e:
                logger.error(f"Error calculating similarity: {e}")
        
        # Fallback to simple token-based similarity
        tokens1 = set(self.tokenize(text1.lower()))
        tokens2 = set(self.tokenize(text2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'nltk_available': NLTK_AVAILABLE,
            'spacy_available': SPACY_AVAILABLE,
            'sentence_transformer_loaded': self.sentence_transformer is not None,
            'bert_model_loaded': self.embedding_model is not None,
            'intent_classifier_loaded': self.intent_classifier is not None,
            'sentiment_analyzer_loaded': self.sentiment_analyzer is not None,
            'ner_pipeline_loaded': self.ner_pipeline is not None,
            'device': str(self.device) if self.device else 'cpu'
        }