import sys
import logging
import os
from pathlib import Path

# Configure logging to a file to capture all output
log_file = Path(__file__).parent / 'simple_test.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    logging.info("--- Starting simple_test.py ---")
    try:
        # Add project root to Python path
        project_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(project_root))
        logging.info(f"Added {project_root} to sys.path")

        # Set GitHub token
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            logging.error("GITHUB_TOKEN environment variable not set.")
            return 1
        os.environ["GITHUB_TOKEN"] = github_token
        logging.info("GITHUB_TOKEN is set.")

        # Import and run the data collection script
        logging.info("Importing collect_data...")
        from scripts.collect_data import main as collect_data_main
        logging.info("Import successful. Running collect_data_main()...")
        collect_data_main()
        logging.info("collect_data_main() finished.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return 1

    logging.info("--- simple_test.py finished successfully ---")
    return 0

if __name__ == "__main__":
    sys.exit(main())
