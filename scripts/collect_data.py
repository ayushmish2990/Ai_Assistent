#!/usr/bin/env python3
"""
Data collection pipeline for the AI Coding Assistant.

This script collects code data from GitHub, generates synthetic examples,
and injects bugs to create a diverse training dataset.
"""

import os
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.data.collectors.github_collector import GitHubCollector
from src.data.collectors.code_extractor import CodeExtractor
from src.data.collectors.synthetic_generator import SyntheticGenerator
from src.data.collectors.bug_generator import BugGenerator


class DataCollector:
    """Main class for collecting and processing code data."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the data collector with configuration."""
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get('output_dir', 'data/collected'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory set to: {self.output_dir}")
        
        # Initialize components
        logging.info("Initializing components...")
        self.github_collector = GitHubCollector(self.config.get('github', {}))
        logging.info("GitHubCollector initialized.")
        self.code_extractor = CodeExtractor()
        logging.info("CodeExtractor initialized.")
        self.synthetic_generator = SyntheticGenerator(
            self.config.get('synthetic', {})
        )
        logging.info("SyntheticGenerator initialized.")
        self.bug_generator = BugGenerator(
            self.config.get('bug_injection', {})
        )
        logging.info("BugGenerator initialized.")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'github': {
                'token': os.getenv('GITHUB_TOKEN'),
                'rate_limit_delay': 1.0,
                'max_retries': 3,
            },
            'search': {
                'languages': ['python', 'javascript'],
                'min_stars': 100,
                'max_repos': 10,
                'has_tests': True,
            },
            'synthetic': {
                'num_samples': 20,
                'languages': ['python', 'javascript'],
                'difficulty_distribution': {
                    'easy': 0.5,
                    'medium': 0.3,
                    'hard': 0.2,
                }
            },
            'bug_injection': {
                'bug_type_weights': {
                    'off_by_one': 0.5,
                    'null_pointer': 0.3,
                    'type_error': 0.2,
                },
                'num_bugs_per_file': 1,
            },
            'output_dir': 'data/collected',
        }
    
    def collect_github_repos(self) -> List[Dict]:
        """Collect repositories from GitHub."""
        logging.info("Collecting GitHub repositories...")
        
        # Get search parameters from config
        search_config = self.config.get('search', {})
        
        # Collect repositories
        repos = self.github_collector.search_repositories(
            languages=search_config.get('languages', ['python']),
            min_stars=search_config.get('min_stars', 100),
            max_repos=search_config.get('max_repos', 10),
            has_tests=search_config.get('has_tests', True),
        )
        
        # Save repositories
        repos_file = self.output_dir / 'github_repos.json'
        with open(repos_file, 'w') as f:
            json.dump([repo.to_dict() for repo in repos], f, indent=2)
        
        logging.info(f"Collected {len(repos)} repositories. Saved to {repos_file}")
        return repos
    
    def extract_code_from_repos(self, repos: List[Dict]) -> List[Dict]:
        """Extract code functions from repositories."""
        logging.info("Extracting code from repositories...")
        
        all_functions = []
        
        for repo in repos:
            try:
                # In a real implementation, we would clone the repo here
                # For now, we'll just use a placeholder
                logging.info(f"Processing repository: {repo.get('full_name', 'unknown')}")
                
                # Simulate finding some files
                # In a real implementation, we would walk the repo directory
                # and process each source file
                file_paths = [
                    f"{repo.get('name', 'repo')}/example.py",
                    f"{repo.get('name', 'repo')}/utils.py",
                ]
                
                for file_path in file_paths:
                    try:
                        # In a real implementation, we would read the file
                        # For now, we'll just create some dummy data
                        if file_path.endswith('.py'):
                            functions = self.code_extractor.extract_functions_from_file(
                                file_path,
                                language='python'
                            )
                            all_functions.extend([f.to_dict() for f in functions])
                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {e}", exc_info=True)
                
            except Exception as e:
                logging.error(f"Error processing repository: {e}", exc_info=True)
        
        # Save extracted functions
        functions_file = self.output_dir / 'extracted_functions.json'
        with open(functions_file, 'w') as f:
            json.dump(all_functions, f, indent=2)
        
        logging.info(f"Extracted {len(all_functions)} functions. Saved to {functions_file}")
        return all_functions
    
    def generate_synthetic_examples(self) -> List[Dict]:
        """Generate synthetic code examples."""
        logging.info("Generating synthetic examples...")
        
        synthetic_config = self.config.get('synthetic', {})
        
        examples = self.synthetic_generator.generate_examples(
            num_examples=synthetic_config.get('num_samples', 20),
            languages=synthetic_config.get('languages', ['python']),
        )
        
        # Convert to dictionaries
        examples_data = [ex.to_dict() for ex in examples]
        
        # Save synthetic examples
        examples_file = self.output_dir / 'synthetic_examples.json'
        with open(examples_file, 'w') as f:
            json.dump(examples_data, f, indent=2)
        
        logging.info(f"Generated {len(examples)} synthetic examples. Saved to {examples_file}")
        return examples_data
    
    def inject_bugs_into_examples(self, examples: List[Dict]) -> List[Dict]:
        """Inject bugs into code examples."""
        logging.info("Injecting bugs into examples...")
        
        bug_config = self.config.get('bug_injection', {})
        buggy_examples = []
        
        for example in examples:
            try:
                code = example.get('code', '')
                if not code.strip():
                    continue
                
                # Inject bugs
                buggy_code, injected_bugs = self.bug_generator.inject_bugs(
                    code=code,
                    language=example.get('language', 'python'),
                    num_bugs=bug_config.get('num_bugs_per_file', 1),
                )
                
                if buggy_code != code:
                    # Create a buggy version of the example
                    buggy_example = example.copy()
                    buggy_example['original_code'] = code
                    buggy_example['code'] = buggy_code
                    buggy_example['injected_bugs'] = [
                        {
                            'bug_type': bug.bug_type,
                            'description': bug.description,
                            'line_number': bug.line_number,
                            'original_code': bug.original_code,
                            'fixed_code': bug.fixed_code,
                        }
                        for bug in injected_bugs
                    ]
                    buggy_examples.append(buggy_example)
            
            except Exception as e:
                logging.error(f"Error injecting bugs into example: {e}", exc_info=True)
        
        # Save buggy examples
        if buggy_examples:
            buggy_file = self.output_dir / 'buggy_examples.json'
            with open(buggy_file, 'w') as f:
                json.dump(buggy_examples, f, indent=2)
            
            logging.info(f"Created {len(buggy_examples)} buggy examples. Saved to {buggy_file}")
        
        return buggy_examples
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete data collection pipeline."""
        logging.info("Starting data collection pipeline...")
        
        # Step 1: Collect GitHub repositories
        repos = self.collect_github_repos()
        
        # Step 2: Extract code from repositories
        functions = self.extract_code_from_repos(repos)
        
        # Step 3: Generate synthetic examples
        synthetic_examples = self.generate_synthetic_examples()
        
        # Step 4: Inject bugs into examples
        buggy_examples = self.inject_bugs_into_examples(synthetic_examples)
        
        # Create a summary of the collected data
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'repositories_collected': len(repos),
            'functions_extracted': len(functions),
            'synthetic_examples_generated': len(synthetic_examples),
            'buggy_examples_created': len(buggy_examples),
            'output_dir': str(self.output_dir.absolute()),
        }
        
        # Save summary
        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info("\nData collection complete!")
        logging.info(f"Summary: {summary_file}")
        
        return summary


def main():
    """Main entry point for the data collection script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Coding Assistant - Data Collection')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (YAML)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for collected data'
    )
    
    args = parser.parse_args()
    
    # Initialize data collector
    logging.info("Initializing DataCollector...")
    collector = DataCollector(config_path=args.config)
    logging.info("DataCollector initialized.")
    
    # Override output directory if specified
    if args.output_dir:
        logging.info(f"Overriding output directory to: {args.output_dir}")
        collector.output_dir = Path(args.output_dir)
        collector.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the pipeline
    logging.info("Starting pipeline run...")
    collector.run_pipeline()
    logging.info("Pipeline run finished.")


if __name__ == '__main__':
    main()
