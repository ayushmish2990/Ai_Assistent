import sys
import os
from pathlib import Path

# Print basic information
print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Current working directory:", os.getcwd())
print("\nPython path:")
for p in sys.path:
    print(f" - {p}")

# Try to import the module directly
print("\nTrying to import github_collector...")
try:
    from src.data.collectors import github_collector
    print("✅ Successfully imported github_collector")
    print(f"Module location: {github_collector.__file__}")
except ImportError as e:
    print(f"❌ Failed to import github_collector: {e}")
    import traceback
    traceback.print_exc()

# Check if the file exists
module_path = Path("src/data/collectors/github_collector.py").resolve()
print(f"\nChecking if file exists: {module_path}")
print(f"File exists: {module_path.exists()}")

# If file exists, try to read it
if module_path.exists():
    print("\nFile contents (first 10 lines):")
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 10:  # Only print first 10 lines
                    print(f"{i+1}: {line.rstrip()}")
                else:
                    break
    except Exception as e:
        print(f"Error reading file: {e}")
