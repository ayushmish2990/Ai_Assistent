# Simple test script to verify Python execution
print("Python is working!")
print(f"Python version: {__import__('sys').version}")
print("Testing basic imports...")
try:
    import requests
    print(f"- requests: {requests.__version__}")
except ImportError:
    print("- requests: Not installed")
    
try:
    import tqdm
    print(f"- tqdm: {tqdm.__version__}")
except ImportError:
    print("- tqdm: Not installed")
    
print("\nEnvironment variables:")
for var in ["PATH", "PYTHONPATH", "GITHUB_TOKEN"]:
    print(f"{var}: {__import__('os').getenv(var, 'Not set')}")
