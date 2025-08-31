import sys
import os

def main():
    info = []
    
    # Basic Python info
    info.append(f"Python Version: {sys.version}")
    info.append(f"Executable: {sys.executable}")
    info.append(f"Working Directory: {os.getcwd()}")
    
    # Environment variables
    info.append("\nEnvironment Variables:")
    for var in ['PATH', 'PYTHONPATH', 'GITHUB_TOKEN']:
        info.append(f"{var}: {os.environ.get(var, 'Not set')}")
    
    # Write to file
    with open('python_environment.txt', 'w') as f:
        f.write('\n'.join(info))
    
    print("Environment information saved to python_environment.txt")

if __name__ == "__main__":
    main()
