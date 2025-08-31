#!/usr/bin/env python3
"""
Terminal Monitor - Watches terminal output for errors and triggers fixes
"""

import asyncio
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Callable
import json
import time
from dataclasses import dataclass
from loguru import logger

@dataclass
class ErrorEvent:
    """Represents a detected error event"""
    timestamp: float
    error_type: str
    language: str
    message: str
    file_path: Optional[str]
    line_number: Optional[int]
    command: str
    working_directory: str
    full_output: str

class TerminalMonitor:
    """Monitors terminal output and detects programming errors"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.error_patterns = self._load_error_patterns()
        self.callbacks: List[Callable[[ErrorEvent], None]] = []
        self.is_monitoring = False
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        default_config = {
            "monitor_interval": 0.1,
            "max_output_length": 10000,
            "supported_languages": ["python", "javascript", "typescript", "java", "cpp"],
            "log_level": "INFO"
        }
        return default_config
    
    def _load_error_patterns(self) -> Dict[str, List[Dict]]:
        """Load error detection patterns for different languages"""
        return {
            "python": [
                {
                    "pattern": r"File \"([^\"]+)\", line (\d+).*\n.*\n(.+Error: .+)",
                    "type": "runtime_error",
                    "groups": {"file": 1, "line": 2, "message": 3}
                },
                {
                    "pattern": r"SyntaxError: (.+)",
                    "type": "syntax_error",
                    "groups": {"message": 1}
                }
            ]
        }
    
    def register_callback(self, callback: Callable[[ErrorEvent], None]):
        """Register a callback to be called when errors are detected"""
        self.callbacks.append(callback)
    
    def detect_language(self, command: str, output: str) -> str:
        """Detect programming language from command and output"""
        if 'python' in command or '.py' in command:
            return 'python'
        elif 'node' in command or 'npm' in command or '.js' in command:
            return 'javascript'
        return 'unknown'
    
    def parse_error(self, output: str, command: str, working_dir: str) -> Optional[ErrorEvent]:
        """Parse output for errors and return ErrorEvent if found"""
        language = self.detect_language(command, output)
        
        if language not in self.error_patterns:
            return None
            
        for pattern_info in self.error_patterns[language]:
            pattern = pattern_info["pattern"]
            match = re.search(pattern, output, re.MULTILINE | re.DOTALL)
            
            if match:
                groups = pattern_info["groups"]
                
                file_path = match.group(groups.get("file", 0)) if "file" in groups else None
                line_number = int(match.group(groups["line"])) if "line" in groups else None
                message = match.group(groups["message"]) if "message" in groups else ""
                
                return ErrorEvent(
                    timestamp=time.time(),
                    error_type=pattern_info["type"],
                    language=language,
                    message=message.strip(),
                    file_path=file_path,
                    line_number=line_number,
                    command=command,
                    working_directory=working_dir,
                    full_output=output
                )
        
        return None
    
    async def start_monitoring(self):
        """Start monitoring terminal processes"""
        self.is_monitoring = True
        logger.info("Terminal monitoring started")
        
        while self.is_monitoring:
            await asyncio.sleep(1)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        logger.info("Terminal monitoring stopped")
