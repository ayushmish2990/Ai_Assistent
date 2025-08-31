"""Synthetic data generator for the AI Coding Assistant."""

import random
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from .code_extractor import CodeFunction


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class SyntheticExample:
    """A synthetic code example with metadata."""
    
    title: str
    description: str
    language: str
    difficulty: str
    code: str
    test_code: Optional[str] = None
    explanation: Optional[str] = None
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.tags is None:
            data['tags'] = []
        return data


class SyntheticGenerator:
    """Generates synthetic code examples for training the AI model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the synthetic data generator."""
        self.config = config or {}
        self._setup_templates()
    
    def _setup_templates(self) -> None:
        """Initialize code templates for different languages and difficulties."""
        self.templates = {
            'python': {
                DifficultyLevel.EASY: self._get_python_easy_templates(),
                DifficultyLevel.MEDIUM: self._get_python_medium_templates(),
                DifficultyLevel.HARD: self._get_python_hard_templates(),
            },
            'javascript': {
                DifficultyLevel.EASY: self._get_javascript_easy_templates(),
                DifficultyLevel.MEDIUM: self._get_javascript_medium_templates(),
                DifficultyLevel.HARD: [],  # Not implemented yet
            },
            'java': {
                DifficultyLevel.EASY: self._get_java_easy_templates(),
                DifficultyLevel.MEDIUM: [],  # Not implemented yet
                DifficultyLevel.HARD: [],  # Not implemented yet
            },
        }
    
    def generate_examples(
        self,
        num_examples: int = 100,
        languages: Optional[List[str]] = None,
        difficulty: Optional[Union[str, DifficultyLevel]] = None,
        **kwargs
    ) -> List[SyntheticExample]:
        """Generate synthetic code examples.
        
        Args:
            num_examples: Number of examples to generate
            languages: List of programming languages to generate examples for
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            
        Returns:
            List of generated examples
        """
        if languages is None:
            languages = self.config.get('languages', ['python'])
        
        if difficulty is None:
            difficulty = self.config.get('difficulty', 'medium')
        
        if isinstance(difficulty, str):
            difficulty = DifficultyLevel(difficulty.lower())
        
        examples = []
        
        for _ in range(num_examples):
            lang = random.choice(languages)
            
            # Get available templates for the selected language and difficulty
            lang_templates = self.templates.get(lang, {})
            templates = lang_templates.get(difficulty, [])
            
            if not templates:
                # Fall back to any available difficulty
                templates = [t for diff in DifficultyLevel for t in lang_templates.get(diff, [])]
            
            if not templates:
                continue
            
            # Select and instantiate a random template
            template = random.choice(templates)
            example = template()
            
            # Add some randomness to the example
            example = self._randomize_example(example, lang)
            examples.append(example)
        
        return examples
    
    def _randomize_example(
        self,
        example: SyntheticExample,
        language: str
    ) -> SyntheticExample:
        """Add randomness to a synthetic example."""
        # This is a simple implementation that can be expanded
        # to add more sophisticated randomization
        
        # Replace placeholders with random values
        if language == 'python':
            example.code = self._randomize_python_code(example.code)
        elif language == 'javascript':
            example.code = self._randomize_javascript_code(example.code)
        
        return example
    
    def _randomize_python_code(self, code: str) -> str:
        """Add randomness to Python code."""
        replacements = {
            '{{number}}': str(random.randint(1, 100)),
            '{{float}}': f"{random.random() * 100:.2f}",
            '{{bool}}': random.choice(['True', 'False']),
            '{{string}}': f"'{random.choice(['apple', 'banana', 'cherry'])}'",
        }
        
        for placeholder, value in replacements.items():
            code = code.replace(placeholder, value)
        
        return code
    
    def _randomize_javascript_code(self, code: str) -> str:
        """Add randomness to JavaScript code."""
        replacements = {
            '{{number}}': str(random.randint(1, 100)),
            '{{float}}': f"{random.random() * 100:.2f}",
            '{{bool}}': random.choice(['true', 'false']),
            '{{string}}': f"'{random.choice(['apple', 'banana', 'cherry'])}'",
        }
        
        for placeholder, value in replacements.items():
            code = code.replace(placeholder, value)
        
        return code
    
    # Template generators for different languages and difficulties
    
    def _get_python_easy_templates(self) -> list:
        """Get easy Python templates."""
        return [
            self._create_python_hello_world,
            self._create_python_add_numbers,
            self._create_python_factorial,
        ]
    
    def _get_python_medium_templates(self) -> list:
        """Get medium Python templates."""
        return [
            self._create_python_fibonacci,
            self._create_python_binary_search,
            self._create_python_sorting_algorithms,
            self._create_python_data_structures,
        ]
    
    def _get_python_hard_templates(self) -> list:
        """Get hard Python templates."""
        return [
            self._create_python_knapsack,
            self._create_python_dijkstra,
        ]
    
    def _get_javascript_easy_templates(self) -> list:
        """Get easy JavaScript templates."""
        return [
            self._create_javascript_hello_world,
            self._create_javascript_add_numbers,
        ]
    
    def _get_javascript_medium_templates(self) -> list:
        """Get medium JavaScript templates."""
        return [
            self._create_javascript_fibonacci,
            self._create_javascript_promise_example,
        ]
    
    def _get_java_easy_templates(self) -> list:
        """Get easy Java templates."""
        return [
            self._create_java_hello_world,
            self._create_java_factorial,
        ]
    
    # Python template functions
    
    def _create_python_hello_world(self) -> SyntheticExample:
        """Create a simple hello world example in Python."""
        code = """def greet(name):
    return f"Hello, {name}!"

# Example usage
print(greet("World"))"""
        
        return SyntheticExample(
            title="Python Hello World",
            description="A simple function that greets a person by name.",
            language="python",
            difficulty=DifficultyLevel.EASY.value,
            code=code,
            test_code="""def test_greet():
    assert greet("Alice") == "Hello, Alice!"
    assert greet("") == "Hello, !"
    print("All tests passed!")

test_greet()""",
            explanation="This function demonstrates string formatting in Python using f-strings.",
            tags=["beginner", "string-formatting", "function"]
        )
    
    def _create_python_add_numbers(self) -> SyntheticExample:
        """Create a function that adds two numbers in Python."""
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        
        code = f"""def add_numbers(a, b):
    return a + b

# Example usage
result = add_numbers({a}, {b})
print(f"{{a}} + {{b}} = {{result}}")"""
        
        return SyntheticExample(
            title="Python Add Numbers",
            description="A function that adds two numbers together.",
            language="python",
            difficulty=DifficultyLevel.EASY.value,
            code=code,
            test_code=f"""def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0
    print("All tests passed!")

test_add_numbers()""",
            explanation="This function demonstrates basic arithmetic operations in Python.",
            tags=["beginner", "arithmetic", "function"]
        )
    
    def _create_python_factorial(self) -> SyntheticExample:
        """Create a factorial function in Python."""
        n = random.randint(1, 10)
        
        code = f"""def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

# Example usage
n = {n}
result = factorial(n)
print(f"{{n}}! = {{result}}")"""
        
        return SyntheticExample(
            title="Python Factorial",
            description="A recursive function to calculate the factorial of a number.",
            language="python",
            difficulty=DifficultyLevel.EASY.value,
            code=code,
            test_code="""def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
    print("All tests passed!")

test_factorial()""",
            explanation="This function demonstrates recursion in Python.",
            tags=["recursion", "math", "function"]
        )
    
    # Add more template functions for different examples...
    
    # JavaScript template functions
    
    def _create_javascript_hello_world(self) -> SyntheticExample:
        """Create a simple hello world example in JavaScript."""
        code = """function greet(name) {
    return `Hello, ${name}!`;
}

// Example usage
console.log(greet("World"));"""
        
        return SyntheticExample(
            title="JavaScript Hello World",
            description="A simple function that greets a person by name.",
            language="javascript",
            difficulty=DifficultyLevel.EASY.value,
            code=code,
            test_code="""function testGreet() {
    console.assert(greet("Alice") === "Hello, Alice!");
    console.assert(greet("") === "Hello, !");
    console.log("All tests passed!");
}

testGreet();""",
            explanation="This function demonstrates template literals in JavaScript.",
            tags=["beginner", "string-formatting", "function"]
        )
    
    # Java template functions
    
    def _create_java_hello_world(self) -> SyntheticExample:
        """Create a simple hello world example in Java."""
        code = """public class HelloWorld {
    public static String greet(String name) {
        return "Hello, " + name + "!";
    }
    
    public static void main(String[] args) {
        // Example usage
        System.out.println(greet("World"));
    }
}"""
        
        return SyntheticExample(
            title="Java Hello World",
            description="A simple method that greets a person by name.",
            language="java",
            difficulty=DifficultyLevel.EASY.value,
            code=code,
            test_code="""public class HelloWorldTest {
    public static void main(String[] args) {
        String result1 = HelloWorld.greet("Alice");
        String result2 = HelloWorld.greet("");
        
        if (!result1.equals("Hello, Alice!")) {
            throw new AssertionError("Test 1 failed");
        }
        if (!result2.equals("Hello, !")) {
            throw new AssertionError("Test 2 failed");
        }
        
        System.out.println("All tests passed!");
    }
}""",
            explanation="This example demonstrates basic Java syntax including class definition, methods, and string concatenation.",
            tags=["beginner", "java", "string-concatenation", "method"]
        )
    
    def _create_python_binary_search(self) -> SyntheticExample:
        """Generate a Python binary search problem."""
        code = '''def binary_search(arr, target):
    """
    Implement binary search to find target in sorted array.
    
    Args:
        arr: Sorted list of integers
        target: Integer to search for
    
    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1'''

        test_code = '''def test_binary_search():
    assert binary_search([1, 3, 5, 7, 9], 5) == 2
    assert binary_search([1, 3, 5, 7, 9], 1) == 0
    assert binary_search([1, 3, 5, 7, 9], 9) == 4
    assert binary_search([1, 3, 5, 7, 9], 4) == -1
    print("All tests passed!")

test_binary_search()'''

        return SyntheticExample(
            title="Binary Search Implementation",
            description="Implement binary search algorithm to find target element in sorted array",
            language="python",
            difficulty=DifficultyLevel.MEDIUM.value,
            code=code,
            test_code=test_code,
            explanation="Hints: Use two pointers (left and right). Calculate the middle index. Compare the middle element with the target and adjust the search range.",
            tags=["algorithm", "binary_search", "divide_and_conquer"]
        )

    def _create_python_sorting_algorithms(self) -> SyntheticExample:
        """Generate a Python quicksort problem."""
        code = '''def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)'''

        test_code = '''def test_quicksort():
    arr = [64, 34, 25, 12, 22, 11, 90]
    quicksort(arr)
    assert arr == [11, 12, 22, 25, 34, 64, 90]
    print("All tests passed!")

test_quicksort()'''

        return SyntheticExample(
            title="Implement Quick Sort",
            description="Implement the quicksort algorithm with proper partitioning",
            language="python",
            difficulty=DifficultyLevel.MEDIUM.value,
            code=code,
            test_code=test_code,
            explanation="This implements the quicksort algorithm using the Lomuto partition scheme.",
            tags=["algorithm", "sorting", "quicksort"]
        )

    def _create_python_data_structures(self) -> SyntheticExample:
        """Generate a Python MinStack problem."""
        code = '''class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        if self.stack:
            val = self.stack.pop()
            if val == self.min_stack[-1]:
                self.min_stack.pop()
            return val
    
    def top(self):
        return self.stack[-1] if self.stack else None
    
    def getMin(self):
        return self.min_stack[-1] if self.min_stack else None'''

        test_code = '''def test_min_stack():
    s = MinStack()
    s.push(-2)
    s.push(0)
    s.push(-3)
    assert s.getMin() == -3
    s.pop()
    assert s.top() == 0
    assert s.getMin() == -2
    print("All tests passed!")

test_min_stack()'''

        return SyntheticExample(
            title="Implement Stack with Min Operation",
            description="Implement a stack that supports push, pop, top, and getMin in O(1) time",
            language="python",
            difficulty=DifficultyLevel.MEDIUM.value,
            code=code,
            test_code=test_code,
            explanation="This implementation uses an auxiliary stack to keep track of the minimum element.",
            tags=["data_structure", "stack", "optimization"]
        )

    # Add more template functions for other algorithms and languages...
