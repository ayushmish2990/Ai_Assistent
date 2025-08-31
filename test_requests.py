import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("\nTrying to import requests...")

try:
    import requests
    print("✅ Successfully imported requests")
    print(f"Requests version: {requests.__version__}")
    
    # Test a simple GET request
    print("\nTesting requests.get()...")
    response = requests.get("https://api.github.com", timeout=5)
    print(f"Status code: {response.status_code}")
    print(f"Response content: {response.text[:200]}...")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
