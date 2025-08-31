import sys
import os
from pathlib import Path

print(f"Current Working Directory: {os.getcwd()}")
print("\n--- sys.path ---")
for p in sys.path:
    print(p)
print("----------------\n")

# Manually add the project root to the path, similar to the setup script
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root added to path: {project_root}")
print("\n--- sys.path after modification ---")
for p in sys.path:
    print(p)
print("----------------\n")

try:
    print("Attempting to import from src.data.collectors.github_collector...")
    from src.data.collectors.github_collector import GitHubCollector
    print("\n✅ SUCCESS: GitHubCollector imported successfully!")
except ImportError as e:
    print(f"\n❌ FAILED: Could not import GitHubCollector.")
    print(f"Error: {e}")
except Exception as e:
    print(f"\n❌ FAILED: An unexpected error occurred.")
    print(f"Error: {e}")
