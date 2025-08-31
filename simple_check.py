import sys
print("Python is working!")
print(f"Python version: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Path: {sys.path}")

# Test basic imports
try:
    import os
    print("✅ os module works")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
except Exception as e:
    print(f"❌ Error with os module: {e}")
