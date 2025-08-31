"""
Base collector class for data collection.

This module defines the base interface for all data collectors.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BaseCollector(ABC):
    """Base class for all data collectors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the collector with configuration.
        
        Args:
            config: Configuration dictionary for the collector
        """
        self.config = config or {}
        self._setup()
    
    def _setup(self) -> None:
        """Setup the collector with configuration."""
        pass
    
    @abstractmethod
    def collect(self, *args, **kwargs) -> Any:
        """Collect data from the source.
        
        Returns:
            Collected data in a format specific to the collector
        """
        pass
    
    def save(self, data: Any, output_path: Union[str, Path]) -> None:
        """Save collected data to a file.
        
        Args:
            data: Data to save
            output_path: Path to save the data to
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.json':
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif output_path.suffix in ('.yaml', '.yml'):
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
    
    def load(self, input_path: Union[str, Path]) -> Any:
        """Load collected data from a file.
        
        Args:
            input_path: Path to load the data from
            
        Returns:
            Loaded data
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if input_path.suffix == '.json':
            import json
            with open(input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif input_path.suffix in ('.yaml', '.yml'):
            import yaml
            with open(input_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def validate(self, data: Any) -> bool:
        """Validate the collected data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        return data is not None
    
    def process(self, data: Any) -> Any:
        """Process the collected data.
        
        Args:
            data: Data to process
            
        Returns:
            Processed data
        """
        return data
    
    def run(self, *args, **kwargs) -> Any:
        """Run the complete collection pipeline.
        
        Returns:
            Collected and processed data
        """
        data = self.collect(*args, **kwargs)
        if self.validate(data):
            return self.process(data)
        raise ValueError("Collected data failed validation")
