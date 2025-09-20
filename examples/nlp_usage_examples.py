"""
NLP Usage Examples for AI Coding Assistant

This module demonstrates practical usage scenarios for the NLP features
integrated into the AI coding assistant. These examples show how developers
can leverage NLP capabilities for various coding tasks.
"""

import asyncio
import sys
import os
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.nlp_engine import NLPEngine, IntentType
from core.nlp_models import NLPModelManager
from core.nlp_capabilities import (
    IntentClassificationCapability,
    CodeUnderstandingCapability,
    SemanticSearchCapability
)
from core.ai_capabilities import AICapabilityRegistry, CapabilityType
from core.model_manager import ModelManager
from core.config import Config


class NLPUsageExamples:
    """Collection of practical NLP usage examples."""
    
    def __init__(self):
        """Initialize NLP components for examples."""
        self.config = Config()
        self.model_manager = ModelManager(self.config)
        self.nlp_engine = NLPEngine()
        self.nlp_model_manager = NLPModelManager()
        self.capability_registry = AICapabilityRegistry(self.model_manager, self.config)
    
    def example_1_intent_based_code_assistance(self):
        """
        Example 1: Intent-based Code Assistance
        
        This example shows how to use intent classification to provide
        contextual code assistance based on user requests.
        """
        print("=" * 60)
        print("Example 1: Intent-based Code Assistance")
        print("=" * 60)
        
        # Get intent classification capability
        intent_capability = self.capability_registry.get_capability("intent_classification")
        
        # Sample user requests
        user_requests = [
            "I need to create a function that sorts a list of numbers",
            "How do I fix this IndexError in my Python code?",
            "Can you explain what this recursive function does?",
            "My code is running slowly, how can I optimize it?",
            "I want to refactor this class to follow SOLID principles"
        ]
        
        for request in user_requests:
            result = intent_capability.execute(request)
            intent = result.get("intent", "unknown")
            confidence = result.get("confidence", 0.0)
            
            print(f"\nUser Request: {request}")
            print(f"Detected Intent: {intent}")
            print(f"Confidence: {confidence:.2f}")
            
            # Provide intent-specific assistance
            if intent == "CODE_GENERATION":
                print("→ Assistance: I'll help you generate the requested code.")
            elif intent == "DEBUGGING":
                print("→ Assistance: Let me analyze your code for potential issues.")
            elif intent == "EXPLANATION":
                print("→ Assistance: I'll explain the code functionality step by step.")
            elif intent == "OPTIMIZATION":
                print("→ Assistance: I'll suggest performance improvements.")
            elif intent == "REFACTORING":
                print("→ Assistance: I'll help restructure your code for better design.")
            else:
                print("→ Assistance: I'll provide general coding help.")
    
    def example_2_intelligent_code_search(self):
        """
        Example 2: Intelligent Code Search
        
        This example demonstrates semantic search capabilities for finding
        relevant code snippets and documentation.
        """
        print("\n" + "=" * 60)
        print("Example 2: Intelligent Code Search")
        print("=" * 60)
        
        # Get semantic search capability
        search_capability = self.capability_registry.get_capability("semantic_search")
        
        # Sample code repository (simulated)
        code_repository = [
            {
                "file": "utils/sorting.py",
                "description": "Implementation of various sorting algorithms including quicksort, mergesort, and heapsort",
                "code": "def quicksort(arr): ..."
            },
            {
                "file": "database/connection.py", 
                "description": "Database connection manager with connection pooling and retry logic",
                "code": "class DatabaseManager: ..."
            },
            {
                "file": "api/authentication.py",
                "description": "User authentication system with JWT tokens and session management",
                "code": "class AuthenticationService: ..."
            },
            {
                "file": "utils/file_processor.py",
                "description": "File processing utilities for CSV, JSON, and XML parsing",
                "code": "def process_csv_file(filename): ..."
            },
            {
                "file": "algorithms/graph.py",
                "description": "Graph algorithms including shortest path, minimum spanning tree, and traversal",
                "code": "def dijkstra_shortest_path(graph, start): ..."
            }
        ]
        
        # Extract descriptions for search
        documents = [item["description"] for item in code_repository]
        
        # Sample search queries
        search_queries = [
            "sorting algorithm implementation",
            "database connection handling",
            "user login and authentication",
            "file parsing utilities",
            "shortest path algorithm"
        ]
        
        for query in search_queries:
            result = search_capability.execute(query, documents, top_k=2)
            
            print(f"\nSearch Query: {query}")
            print("Top Results:")
            
            for i, match in enumerate(result["results"][:2]):
                idx = match["index"]
                score = match["score"]
                file_info = code_repository[idx]
                
                print(f"  {i+1}. {file_info['file']} (Score: {score:.3f})")
                print(f"     {file_info['description']}")
    
    def example_3_code_understanding_and_analysis(self):
        """
        Example 3: Code Understanding and Analysis
        
        This example shows how to analyze and understand code structure,
        complexity, and functionality using NLP techniques.
        """
        print("\n" + "=" * 60)
        print("Example 3: Code Understanding and Analysis")
        print("=" * 60)
        
        # Get code understanding capability
        code_capability = self.capability_registry.get_capability("code_understanding")
        
        # Sample code snippets for analysis
        code_samples = [
            {
                "name": "Fibonacci Function",
                "language": "python",
                "code": """
def fibonacci(n):
    '''Calculate the nth Fibonacci number using recursion.'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
                """
            },
            {
                "name": "Binary Search",
                "language": "python", 
                "code": """
def binary_search(arr, target):
    '''Binary search implementation with O(log n) complexity.'''
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
                """
            },
            {
                "name": "User Class",
                "language": "python",
                "code": """
class User:
    '''User model with authentication and profile management.'''
    
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.is_active = True
        self.created_at = datetime.now()
    
    def authenticate(self, password):
        return check_password_hash(self.password_hash, password)
    
    def update_profile(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                """
            }
        ]
        
        for sample in code_samples:
            result = code_capability.execute(sample["code"], sample["language"])
            
            print(f"\nAnalyzing: {sample['name']}")
            print(f"Language: {result.get('language', 'unknown')}")
            print(f"Functions: {', '.join(result.get('functions', []))}")
            print(f"Classes: {', '.join(result.get('classes', []))}")
            print(f"Complexity: {result.get('complexity', 'unknown')}")
            print(f"Purpose: {result.get('purpose', 'Not determined')}")
    
    def example_4_multilingual_code_support(self):
        """
        Example 4: Multilingual Code Support
        
        This example demonstrates NLP capabilities working with different
        programming languages and natural languages.
        """
        print("\n" + "=" * 60)
        print("Example 4: Multilingual Code Support")
        print("=" * 60)
        
        # Multi-language code samples
        multilingual_samples = [
            {
                "language": "python",
                "description": "Python list comprehension example",
                "code": "squares = [x**2 for x in range(10)]"
            },
            {
                "language": "javascript", 
                "description": "JavaScript arrow function example",
                "code": "const multiply = (a, b) => a * b;"
            },
            {
                "language": "java",
                "description": "Java class definition example", 
                "code": "public class Calculator { public int add(int a, int b) { return a + b; } }"
            },
            {
                "language": "cpp",
                "description": "C++ template function example",
                "code": "template<typename T> T max(T a, T b) { return (a > b) ? a : b; }"
            }
        ]
        
        code_capability = self.capability_registry.get_capability("code_understanding")
        
        for sample in multilingual_samples:
            result = code_capability.execute(sample["code"], sample["language"])
            
            print(f"\nLanguage: {sample['language'].upper()}")
            print(f"Description: {sample['description']}")
            print(f"Code: {sample['code']}")
            print(f"Detected Language: {result.get('language', 'unknown')}")
            print(f"Analysis: {result.get('summary', 'No analysis available')}")
    
    def example_5_context_aware_suggestions(self):
        """
        Example 5: Context-aware Code Suggestions
        
        This example shows how NLP can provide context-aware suggestions
        based on code analysis and user intent.
        """
        print("\n" + "=" * 60)
        print("Example 5: Context-aware Code Suggestions")
        print("=" * 60)
        
        # Scenarios with context and user requests
        scenarios = [
            {
                "context": "Working on a web API with Flask",
                "current_code": """
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/users', methods=['GET'])
def get_users():
    # TODO: Implement user retrieval
    pass
                """,
                "user_request": "I need to add error handling to this endpoint"
            },
            {
                "context": "Data processing pipeline",
                "current_code": """
import pandas as pd

def process_data(filename):
    df = pd.read_csv(filename)
    # Basic processing
    df = df.dropna()
    return df
                """,
                "user_request": "How can I make this more efficient for large files?"
            },
            {
                "context": "Machine learning model training",
                "current_code": """
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
                """,
                "user_request": "I want to add cross-validation to improve model evaluation"
            }
        ]
        
        intent_capability = self.capability_registry.get_capability("intent_classification")
        code_capability = self.capability_registry.get_capability("code_understanding")
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\nScenario {i}: {scenario['context']}")
            print(f"User Request: {scenario['user_request']}")
            
            # Analyze user intent
            intent_result = intent_capability.execute(scenario['user_request'])
            intent = intent_result.get("intent", "unknown")
            
            # Analyze current code
            code_result = code_capability.execute(scenario['current_code'], "python")
            
            print(f"Detected Intent: {intent}")
            print(f"Code Analysis: {code_result.get('summary', 'No analysis')}")
            
            # Provide context-aware suggestions
            if intent == "ERROR_HANDLING":
                print("💡 Suggestions:")
                print("  - Add try-catch blocks around risky operations")
                print("  - Implement proper HTTP status codes")
                print("  - Add input validation")
            elif intent == "OPTIMIZATION":
                print("💡 Suggestions:")
                print("  - Use chunking for large file processing")
                print("  - Consider using Dask for parallel processing")
                print("  - Implement memory-efficient data loading")
            elif intent == "TESTING" or "validation" in scenario['user_request'].lower():
                print("💡 Suggestions:")
                print("  - Implement k-fold cross-validation")
                print("  - Add metrics evaluation (precision, recall, F1)")
                print("  - Consider stratified sampling for imbalanced data")
    
    def example_6_automated_documentation(self):
        """
        Example 6: Automated Documentation Generation
        
        This example demonstrates using NLP to generate documentation
        and comments for code automatically.
        """
        print("\n" + "=" * 60)
        print("Example 6: Automated Documentation Generation")
        print("=" * 60)
        
        # Code samples that need documentation
        undocumented_code = [
            """
def calculate_compound_interest(principal, rate, time, n=1):
    amount = principal * (1 + rate/n) ** (n * time)
    return amount - principal
            """,
            """
class DataValidator:
    def __init__(self, rules):
        self.rules = rules
    
    def validate(self, data):
        errors = []
        for rule in self.rules:
            if not rule.check(data):
                errors.append(rule.error_message)
        return errors
            """,
            """
def merge_sorted_arrays(arr1, arr2):
    result = []
    i = j = 0
    
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result
            """
        ]
        
        code_capability = self.capability_registry.get_capability("code_understanding")
        
        for i, code in enumerate(undocumented_code, 1):
            print(f"\nCode Sample {i}:")
            print("Original Code:")
            print(code.strip())
            
            # Analyze code to generate documentation
            result = code_capability.execute(code, "python")
            
            print("\nGenerated Documentation:")
            print(f"Purpose: {result.get('purpose', 'Function/class implementation')}")
            print(f"Parameters: {result.get('parameters', 'Not specified')}")
            print(f"Returns: {result.get('returns', 'Not specified')}")
            print(f"Complexity: {result.get('complexity', 'Not analyzed')}")
            
            # Generate docstring suggestion
            functions = result.get('functions', [])
            if functions:
                print(f"\nSuggested docstring for '{functions[0]}':")
                print('"""')
                print(f"{result.get('purpose', 'Function implementation')}")
                print("")
                print("Args:")
                print("    principal: Initial amount of money")
                print("    rate: Interest rate per period")
                print("    time: Number of periods")
                print("    n: Number of times interest is compounded per period")
                print("")
                print("Returns:")
                print("    float: Compound interest earned")
                print('"""')


def main():
    """Run all NLP usage examples."""
    print("AI Coding Assistant - NLP Usage Examples")
    print("========================================")
    
    examples = NLPUsageExamples()
    
    try:
        # Run all examples
        examples.example_1_intent_based_code_assistance()
        examples.example_2_intelligent_code_search()
        examples.example_3_code_understanding_and_analysis()
        examples.example_4_multilingual_code_support()
        examples.example_5_context_aware_suggestions()
        examples.example_6_automated_documentation()
        
        print("\n" + "=" * 60)
        print("All NLP examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Note: Some examples may require additional setup or dependencies.")


if __name__ == "__main__":
    main()