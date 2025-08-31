"""
Development setup script for the AI Coding Assistant.

This script sets up the development environment by:
1. Creating a virtual environment
2. Installing development dependencies
3. Installing the package in development mode
"""

import os
import sys
import subprocess
import venv
from pathlib import Path
from typing import Optional

def run_command(command: str, cwd: Optional[Path] = None) -> None:
    """Run a shell command."""
    print(f"Running: {command}")
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=cwd or Path.cwd(),
            executable=sys.executable,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    # Project root directory
    project_root = Path(__file__).parent
    venv_dir = project_root / "venv"
    
    # Create virtual environment if it doesn't exist
    if not venv_dir.exists():
        print("Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)
    
    # Determine the correct pip and python paths
    if os.name == 'nt':  # Windows
        pip_path = venv_dir / "Scripts" / "pip"
        python_path = venv_dir / "Scripts" / "python"
    else:  # Unix/Linux/Mac
        pip_path = venv_dir / "bin" / "pip"
        python_path = venv_dir / "bin" / "python"
    
    # Upgrade pip
    print("\nUpgrading pip...")
    run_command(f"{pip_path} install --upgrade pip")
    
    # Install requirements
    print("\nInstalling requirements...")
    run_command(f"{pip_path} install -r requirements.txt")
    
    # Install package in development mode
    print("\nInstalling package in development mode...")
    run_command(f"{pip_path} install -e .")
    
    # Install development dependencies
    print("\nInstalling development dependencies...")
    run_command(f"{pip_path} install -e '.[dev]'")
    
    print("\nSetup complete!")
    print(f"\nTo activate the virtual environment, run:")
    if os.name == 'nt':  # Windows
        print(f"  .\\venv\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print(f"  source venv/bin/activate")
    print("\nTo run tests:")
    print(f"  {python_path} -m pytest")

if __name__ == "__main__":
    main()
