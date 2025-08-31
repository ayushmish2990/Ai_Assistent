"""Verify the Python environment and required packages."""
import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is 3.6 or higher."""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version < (3, 6):
        print("❌ Python 3.6 or higher is required")
        return False
    return True

def check_package(package_name):
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        version = sys.modules[package_name].__version__
        print(f"✅ {package_name} {version} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def install_package(package_name):
    """Install a Python package using pip."""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def main():
    """Main function to verify environment."""
    print("Verifying environment...\n")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check required packages
    required_packages = ["requests", "esprima", "tqdm", "pyyaml"]
    all_installed = True
    
    for package in required_packages:
        if not check_package(package):
            all_installed = False
    
    if not all_installed:
        print("\nSome required packages are missing. Installing...\n")
        for package in required_packages:
            if not check_package(package):
                install_package(package)
    
    # Verify GitHub token
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        print("\n✅ GITHUB_TOKEN is set")
    else:
        print("\n❌ GITHUB_TOKEN is not set")
        print("Please set your GitHub token with:")
        print("  Windows: set GITHUB_TOKEN=your_token_here")
        print("  Unix/Mac: export GITHUB_TOKEN=your_token_here")
    
    print("\nEnvironment verification complete!")
    return all_installed

if __name__ == "__main__":
    main()
