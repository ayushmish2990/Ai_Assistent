import sys
import os

# Open output file
with open('import_test_output.txt', 'w', encoding='utf-8') as f:
    def log(msg):
        print(msg)
        f.write(msg + '\n')
        f.flush()

    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    log(f"Project root: {project_root}")
    log("Testing imports...")

# Test basic Python imports
try:
    import requests
    log("✅ requests")
except ImportError as e:
    print(f"❌ requests: {e}")

try:
    import torch
    log(f"✅ torch {torch.__version__}")
except ImportError as e:
    print(f"❌ torch: {e}")

try:
    import transformers
    log(f"✅ transformers {transformers.__version__}")
except ImportError as e:
    print(f"❌ transformers: {e}")

# Try importing the collect_data module
try:
    log("\nTrying to import collect_data...")
    from scripts.collect_data import main
    log("✅ Successfully imported collect_data")
except Exception as e:
    log(f"❌ Error importing collect_data: {e}")
    import traceback
    tb = traceback.format_exc()
    log(tb)

    log("\nPython path:")
    for p in sys.path:
        log(f" - {p}")

    log("\nEnvironment variables:")
    for var in ["PATH", "PYTHONPATH", "GITHUB_TOKEN"]:
        log(f"{var}: {os.environ.get(var, 'Not set')}")
