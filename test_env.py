import sys
import os

def main():
    with open('test_output.txt', 'w') as f:
        f.write("=== Python Environment Test ===\n")
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"Executable: {sys.executable}\n")
        f.write(f"Working Directory: {os.getcwd()}\n")
        
        f.write("\n=== Environment Variables ===\n")
        for var in ['PATH', 'PYTHONPATH', 'GITHUB_TOKEN']:
            f.write(f"{var}: {os.environ.get(var, 'Not set')}\n")
        
        f.write("\n=== File System Test ===\n")
        try:
            with open('test_write.txt', 'w') as test_file:
                test_file.write("Test write operation")
            f.write("✅ Successfully wrote to test_write.txt\n")
            os.remove('test_write.txt')
        except Exception as e:
            f.write(f"❌ File write test failed: {e}\n")
        
        f.write("\n=== Module Import Test ===\n")
        for module in ['os', 'sys', 'requests', 'tqdm']:
            try:
                __import__(module)
                f.write(f"✅ {module} imported successfully\n")
            except ImportError as e:
                f.write(f"❌ {module} import failed: {e}\n")

if __name__ == "__main__":
    main()
