#!/usr/bin/env python3
"""
Error Parser and Code Context Extractor
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

@dataclass
class CodeContext:
    """Contains code context around an error"""
    file_path: str
    error_line: int
    surrounding_lines: List[Tuple[int, str]]
    function_context: Optional[str]
    class_context: Optional[str]
    imports: List[str]
    variables_in_scope: List[str]
    file_content: str

@dataclass
class ErrorAnalysis:
    """Complete analysis of an error"""
    error_type: str
    error_message: str
    suggested_fixes: List[str]
    context: CodeContext
    confidence: float
    fix_difficulty: str

class ErrorParser:
    """Analyzes errors and extracts context for AI processing"""
    
    def __init__(self):
        self.common_fixes = {
            "NameError": [
                "Check if the variable is defined before use",
                "Verify variable spelling and case sensitivity"
            ],
            "SyntaxError": [
                "Check for missing colons, parentheses, or brackets",
                "Verify proper indentation"
            ]
        }
    
    def extract_code_context(self, file_path: str, error_line: int, context_lines: int = 10) -> Optional[CodeContext]:
        """Extract code context around an error location"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start_line = max(1, error_line - context_lines)
            end_line = min(len(lines), error_line + context_lines)
            
            surrounding_lines = []
            for i in range(start_line - 1, end_line):
                surrounding_lines.append((i + 1, lines[i].rstrip()))
            
            return CodeContext(
                file_path=file_path,
                error_line=error_line,
                surrounding_lines=surrounding_lines,
                function_context=None,
                class_context=None,
                imports=[],
                variables_in_scope=[],
                file_content="\n".join([line[1] for line in surrounding_lines])
            )
            
        except Exception as e:
            logger.error(f"Error extracting code context: {e}")
            return None
    
    def analyze_error(self, error_event, language: str) -> Optional[ErrorAnalysis]:
        """Main method to analyze an error and return analysis"""
        if not error_event.file_path:
            return None
        
        context = self.extract_code_context(
            error_event.file_path, 
            error_event.line_number or 1
        )
        
        if not context:
            return None
        
        error_type = error_event.error_type
        suggested_fixes = self.common_fixes.get(error_type, ["Review the error message"])
        
        return ErrorAnalysis(
            error_type=error_type,
            error_message=error_event.message,
            suggested_fixes=suggested_fixes,
            context=context,
            confidence=0.8,
            fix_difficulty="medium"
        )
