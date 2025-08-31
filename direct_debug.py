import sys
import os

def main():
    # Basic environment info
    print("=== Python Environment ===")
    print(f"Python Version: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check if we can write to a file
    print("\n=== File System Access ===")
    test_file = "test_write.txt"
    try:
        with open(test_file, 'w') as f:
            f.write("Test write operation")
        print(f"✅ Successfully wrote to {test_file}")
        os.remove(test_file)
    except Exception as e:
        print(f"❌ Failed to write to file: {e}")
    
    # Test imports
    print("\n=== Module Imports ===")
    for module in ['os', 'sys', 'requests', 'tqdm']:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError:
            print(f"❌ {module} import failed")
    
    # Test network access
    print("\n=== Network Access ===")
    try:
        import urllib.request
        with urllib.request.urlopen('https://www.python.org', timeout=5) as response:
            print(f"✅ Successfully connected to python.org (Status: {response.status})")
    except Exception as e:
        print(f"❌ Network test failed: {e}")

if __name__ == "__main__":
    main()
