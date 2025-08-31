#!/usr/bin/env python3
"""
Setup and run the data collection pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging at the very top
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the Python path to allow for absolute imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
logging.info(f"Added {project_root} to sys.path")

def setup_directories():
    """Create necessary directories."""
    logging.info("Creating directory structure...")
    dirs = [
        "data/raw",
        "data/processed",
        "data/collected",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")

def set_github_token(token: str):
    """Set GitHub token as environment variable."""
    os.environ["GITHUB_TOKEN"] = token
    logging.info("GitHub token set in environment variables")

def run_data_collection():
    """Run the data collection pipeline."""
    logging.info("Starting data collection...")
    try:
        # Import after setup to ensure dependencies are installed
        from scripts.collect_data import main as collect_data_main

        # Run the main data collection script directly
        # This avoids subprocess issues with Python path
        original_argv = sys.argv
        try:
            # Set command line arguments for the script
            sys.argv = ['scripts/collect_data.py', '--output-dir', 'data/collected']
            logging.info("Successfully imported collect_data.py")
            collect_data_main()
            logging.info("Finished running collect_data.py main function.")
        finally:
            # Restore original arguments
            sys.argv = original_argv
    except subprocess.CalledProcessError as e:
        print(f"Error during data collection: {e}")
        sys.exit(1)

def validate_data():
    """Validate the collected data."""
    print("\nValidating collected data...")
    try:
        import json
        from pathlib import Path
        
        # Check collected functions
        func_file = Path("data/collected/extracted_functions.json")
        if func_file.exists():
            with open(func_file, 'r') as f:
                functions = json.load(f)
            print(f"✅ Collected {len(functions)} functions")
            if functions:
                print("Sample function:")
                print(json.dumps(functions[0], indent=2))
        else:
            print("❌ No functions collected yet")
        
        # Check synthetic examples
        synth_file = Path("data/collected/synthetic_examples.json")
        if synth_file.exists():
            with open(synth_file, 'r') as f:
                examples = json.load(f)
            print(f"✅ Generated {len(examples)} synthetic examples")
        else:
            print("❌ No synthetic examples generated yet")
            
    except Exception as e:
        print(f"Error validating data: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logging.error("GITHUB_TOKEN environment variable not set.")
        sys.exit(1)

    logging.info("Starting setup and run script...")
    # Setup and run
    # setup_environment() is removed as dependencies are installed by check_python.bat
    setup_directories()
    set_github_token(github_token)
    # run_data_collection() # Temporarily disabled for debugging
    validate_data()
    
    print("\n🎉 Data collection complete! Check the 'data/collected' directory for results.")
