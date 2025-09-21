#!/usr/bin/env python3
"""
AI-Powered Code Fixer
"""

import os
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

@dataclass
class FixSuggestion:
    """Represents a suggested fix for code"""
    fix_type: str
    description: str
    original_code: str
    fixed_code: str
    confidence: float
    explanation: str
    file_path: str
    line_range: Tuple[int, int]

@dataclass
class FixResult:
    """Result of applying a fix"""
    success: bool
    message: str
    applied_fix: Optional[FixSuggestion]
    backup_path: Optional[str]

class CodeFixer:
    """AI-powered code fixer"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.ai_provider = self.config.get('ai_provider', 'openai')
        self.model = self.config.get('model', 'gpt-4')
        
        # Initialize AI client
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            self.ai_client = openai.OpenAI()
            logger.info("OpenAI client initialized")
        else:
            logger.warning("OpenAI not available - using rule-based fixes only")
            self.ai_client = None
    
    async def generate_fix(self, error_analysis) -> Optional[FixSuggestion]:
        """Generate a fix suggestion"""
        # Simple rule-based fix for demo
        context = error_analysis.context
        error_type = error_analysis.error_type
        
        if error_type == "NameError" and "is not defined" in error_analysis.error_message:
            # Try to fix variable name typos
            for line_num, line_content in context.surrounding_lines:
                if line_num == context.error_line:
                    # Simple typo fix (this is just a demo)
                    if "nam" in line_content and "name" in str(context.surrounding_lines):
                        fixed_line = line_content.replace("nam", "name")
                        
                        return FixSuggestion(
                            fix_type="rule_based",
                            description="Fix variable name typo",
                            original_code=line_content,
                            fixed_code=fixed_line,
                            confidence=0.9,
                            explanation="Replaced 'nam' with 'name'",
                            file_path=context.file_path,
                            line_range=(context.error_line, context.error_line)
                        )
        
        return None
    
    def show_diff(self, fix_suggestion: FixSuggestion) -> str:
        """Show a diff of the proposed changes"""
        return f"- {fix_suggestion.original_code}\n+ {fix_suggestion.fixed_code}"

class CodeApplier:
    """Applies code fixes to files."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.backup_dir = Path(self.config.get('backup_dir', './backups'))
        self.backup_dir.mkdir(exist_ok=True)

    def create_backup(self, file_path: str) -> str:
        """Create a backup of the file before modification"""
        try:
            file_path = Path(file_path)
            timestamp = int(asyncio.get_event_loop().time())
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}.backup"
            backup_path = self.backup_dir / backup_name
            
            import shutil
            shutil.copy2(file_path, backup_path)
            
            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    async def apply_fix(self, fix_suggestion: FixSuggestion, dry_run: bool = False) -> FixResult:
        """Apply a fix suggestion to the file"""
        if dry_run:
            return FixResult(
                success=True,
                message="Dry run - no changes made",
                applied_fix=fix_suggestion,
                backup_path=None
            )
        
        try:
            # Create backup
            backup_path = self.create_backup(fix_suggestion.file_path)
            
            # Read and modify file
            with open(fix_suggestion.file_path, 'r') as f:
                lines = f.readlines()
            
            # Simple line replacement
            start_line = fix_suggestion.line_range[0] - 1
            lines[start_line] = fix_suggestion.fixed_code + '\n'
            
            # Write back
            with open(fix_suggestion.file_path, 'w') as f:
                f.writelines(lines)
            
            return FixResult(
                success=True,
                message=f"Fix applied. Backup: {backup_path}",
                applied_fix=fix_suggestion,
                backup_path=backup_path
            )
            
        except Exception as e:
            return FixResult(
                success=False,
                message=f"Failed to apply fix: {e}",
                applied_fix=None,
                backup_path=None
            )
