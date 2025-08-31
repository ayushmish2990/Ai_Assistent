import sys
from pathlib import Path

# Add the project root to the Python path
root_dir = str(Path(__file__).parent.resolve())
sys.path.insert(0, root_dir)
