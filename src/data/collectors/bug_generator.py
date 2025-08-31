"""Bug injection utilities for the AI Coding Assistant."""

import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union


class BugType(Enum):
    """Types of bugs that can be injected."""
    OFF_BY_ONE = "off_by_one"
    NULL_POINTER = "null_pointer"
    TYPE_ERROR = "type_error"


@dataclass
class InjectedBug:
    """Represents a bug that has been injected into code."""
    bug_type: str
    description: str
    line_number: int
    original_code: str
    buggy_code: str
    fixed_code: Optional[str] = None


class BugGenerator:
    """Injects various types of bugs into code."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.bug_weights = {
            BugType.OFF_BY_ONE: 0.5,
            BugType.NULL_POINTER: 0.3,
            BugType.TYPE_ERROR: 0.2,
        }
    
    def inject_bugs(
        self,
        code: str,
        language: str = "python",
        num_bugs: int = 1,
    ) -> Tuple[str, List[InjectedBug]]:
        """Inject bugs into the given code."""
        if not code.strip():
            return code, []
        
        lines = code.splitlines()
        injected_bugs = []
        
        # Get non-empty, non-comment lines
        potential_lines = [
            i for i, line in enumerate(lines)
            if line.strip() and not line.strip().startswith(('#', '//', '/*', '*'))
        ]
        
        if not potential_lines:
            return code, []
        
        # Select random lines to inject bugs
        bug_locations = random.sample(
            potential_lines,
            min(num_bugs, len(potential_lines))
        )
        
        # Sort in reverse order to avoid line number issues
        bug_locations.sort(reverse=True)
        
        for line_num in bug_locations:
            if line_num >= len(lines):
                continue
                
            bug_type = random.choices(
                list(self.bug_weights.keys()),
                weights=list(self.bug_weights.values()),
                k=1
            )[0]
            
            original_line = lines[line_num]
            buggy_line, bug_info = self._inject_bug(original_line, bug_type, language)
            
            if buggy_line != original_line:
                injected_bug = InjectedBug(
                    bug_type=bug_type.value,
                    description=bug_info.get('description', 'Unknown bug'),
                    line_number=line_num + 1,
                    original_code=original_line,
                    buggy_code=buggy_line,
                    fixed_code=bug_info.get('fixed_code', original_line),
                )
                lines[line_num] = buggy_line
                injected_bugs.append(injected_bug)
        
        return '\n'.join(lines), injected_bugs
    
    def _inject_bug(self, line: str, bug_type: BugType, language: str) -> Tuple[str, Dict]:
        """Inject a specific type of bug into a line of code."""
        if bug_type == BugType.OFF_BY_ONE:
            return self._inject_off_by_one(line, language)
        elif bug_type == BugType.NULL_POINTER:
            return self._inject_null_pointer(line, language)
        elif bug_type == BugType.TYPE_ERROR:
            return self._inject_type_error(line, language)
        return line, {}
    
    def _inject_off_by_one(self, line: str, language: str) -> Tuple[str, Dict]:
        """Inject an off-by-one error."""
        if 'range(' in line and ')' in line:
            # Simple case: range(0, 10) -> range(1, 10) or range(0, 9)
            parts = line.split('range')
            for i in range(1, len(parts)):
                if '(' in parts[i] and ')' in parts[i]:
                    args = parts[i].split('(')[1].split(')')[0].split(',')
                    if len(args) >= 2 and all(a.strip().lstrip('-').isdigit() for a in args[:2] if a.strip()):
                        # Modify the end of range
                        start = int(args[0].strip())
                        end = int(args[1].strip())
                        new_end = end - 1 if end > start else end + 1
                        new_args = f"{start}, {new_end}" + (',' + ','.join(args[2:]) if len(args) > 2 else '')
                        new_line = line.replace(
                            f"range({','.join(args)}",
                            f"range({new_args}"
                        )
                        return new_line, {
                            'description': 'Off-by-one error in range()',
                            'fixed_code': line
                        }
        return line, {}
    
    def _inject_null_pointer(self, line: str, language: str) -> Tuple[str, Dict]:
        """Inject a null pointer dereference."""
        if ' = ' in line and not any(x in line for x in ['None', 'null', 'NULL']):
            # Add a potential null assignment
            parts = line.split(' = ')
            if len(parts) == 2 and not parts[1].strip().startswith('None'):
                new_line = f"{parts[0]} = None  # Potential null pointer"
                return new_line, {
                    'description': 'Potential null pointer assignment',
                    'fixed_code': line
                }
        return line, {}
    
    def _inject_type_error(self, line: str, language: str) -> Tuple[str, Dict]:
        """Inject a type error."""
        if ' + ' in line and not any(x in line for x in ['str', 'int(']):
            # Add a string + number operation
            parts = line.split(' + ')
            if len(parts) == 2:
                # This logic is flawed, replacing with a simpler version for now
                # The original was trying to randomly add str() to one side of a '+'
                # but was syntactically incorrect.
                new_line = line.replace(' + ', ' + "" + ')
                return new_line, {
                    'description': 'Potential type error in addition',
                    'fixed_code': line
                }
        return line, {}
