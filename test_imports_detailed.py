import sys
import os
import logging
from pathlib import Path

# Set up logging to file
log_file = Path(__file__).parent / 'import_test.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

def log(msg):
    print(msg)
    logging.info(msg)

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
log(f"Project root: {project_root}")
log(f"Python path: {sys.path}")

# Log environment variables
log("\nEnvironment variables:")
for var in ['PATH', 'PYTHONPATH', 'GITHUB_TOKEN']:
    log(f"{var}: {os.environ.get(var, 'Not set')}")

# List files in project root
log("\nFiles in project root:")
for f in project_root.glob('*'):
    log(f" - {f.name}")

# List files in src directory
src_dir = project_root / 'src'
if src_dir.exists():
    log(f"\nContents of {src_dir}:")
    for f in src_dir.glob('**/*.py'):
        log(f" - {f.relative_to(project_root)}")
else:
    log(f"\nError: {src_dir} does not exist")

# Test imports
log("\nTesting imports from src.data.collectors:")

try:
    log("\nAttempting to import GitHubCollector...")
    from src.data.collectors.github_collector import GitHubCollector
    log("✅ Successfully imported GitHubCollector")
    
    log("\nAttempting to import CodeExtractor...")
    from src.data.collectors.code_extractor import CodeExtractor
    log("✅ Successfully imported CodeExtractor")
    
    log("\nAttempting to import SyntheticGenerator...")
    from src.data.collectors.synthetic_generator import SyntheticGenerator
    log("✅ Successfully imported SyntheticGenerator")
    
    log("\nAttempting to import BugGenerator...")
    from src.data.collectors.bug_generator import BugGenerator
    log("✅ Successfully imported BugGenerator")
    
    log("\n✅ All imports successful!")
    
except ImportError as e:
    log(f"❌ ImportError: {e}")
    log("\nPython path:")
    for p in sys.path:
        log(f" - {p}")
        
    # Check if the module files exist
    module_path = project_root / 'src' / 'data' / 'collectors'
    log(f"\nChecking for module files in {module_path}:")
    if module_path.exists():
        for f in module_path.glob('*.py'):
            log(f" - {f.name} (exists)")
    else:
        log(f"Module directory not found: {module_path}")
        
    # Show the full traceback
    import traceback
    log("\nFull traceback:")
    log(traceback.format_exc())
    
except Exception as e:
    log(f"❌ Unexpected error: {e}")
    import traceback
    log("\nFull traceback:")
    log(traceback.format_exc())
