#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Coding Assistant - Main Application
"""

import asyncio
import sys
import os
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitors.terminal_monitor import TerminalMonitor, ErrorEvent
from parsers.error_parser import ErrorParser
from fixers.code_fixer import CodeFixer

class AICodingAssistant:
    """Main application that coordinates all components"""
    
    def __init__(self):
        self.terminal_monitor = TerminalMonitor()
        self.error_parser = ErrorParser()
        self.code_fixer = CodeFixer()
        
        self.stats = {
            'errors_detected': 0,
            'fixes_generated': 0,
            'fixes_applied': 0
        }
        
        # Register error handler
        self.terminal_monitor.register_callback(self._handle_error_event)
        
        logger.info("AI Coding Assistant initialized")
    
    async def _handle_error_event(self, error_event: ErrorEvent):
        """Handle detected error events"""
        try:
            self.stats['errors_detected'] += 1
            
            print(f"\n��� Error detected: {error_event.error_type}")
            print(f"��� File: {error_event.file_path}:{error_event.line_number}")
            print(f"��� Message: {error_event.message}")
            
            # Analyze the error
            error_analysis = self.error_parser.analyze_error(error_event, error_event.language)
            
            if not error_analysis:
                print("❌ Could not analyze error")
                return
            
            print(f"��� Analysis: {error_analysis.error_type} (confidence: {error_analysis.confidence:.2f})")
            
            # Generate fix suggestion
            fix_suggestion = await self.code_fixer.generate_fix(error_analysis)
            
            if not fix_suggestion:
                print("❌ Could not generate fix suggestion")
                return
            
            self.stats['fixes_generated'] += 1
            print(f"��� Fix suggested: {fix_suggestion.description}")
            print(f"��� Confidence: {fix_suggestion.confidence:.2f}")
            
            # Show diff
            print("\n��� Proposed changes:")
            print(self.code_fixer.show_diff(fix_suggestion))
            
            # Ask for confirmation
            response = input("\n❓ Apply this fix? [y/n]: ").lower().strip()
            
            if response in ['y', 'yes']:
                result = await self.code_fixer.apply_fix(fix_suggestion)
                if result.success:
                    self.stats['fixes_applied'] += 1
                    print(f"✅ {result.message}")
                else:
                    print(f"❌ {result.message}")
            else:
                print("⏭️  Fix skipped")
                
        except Exception as e:
            logger.error(f"Error handling error event: {e}")
    
    async def start(self):
        """Start the AI coding assistant"""
        print("\n" + "="*60)
        print("��� AI CODING ASSISTANT STARTED")
        print("="*60)
        print("��� Monitoring for programming errors...")
        print("��� Ready to suggest fixes")
        print("⚠️  Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        try:
            await self.terminal_monitor.start_monitoring()
        except KeyboardInterrupt:
            await self.stop()
    
    async def stop(self):
        """Stop the AI coding assistant"""
        print("\n��� Stopping AI Coding Assistant...")
        self.terminal_monitor.stop_monitoring()
        
        print(f"\n�� Session Statistics:")
        print(f"��� Errors detected: {self.stats['errors_detected']}")
        print(f"��� Fixes generated: {self.stats['fixes_generated']}")
        print(f"✅ Fixes applied: {self.stats['fixes_applied']}")
        
        print("\n��� Goodbye!")

async def test_mode():
    """Test mode with simulated errors"""
    print("��� Running in test mode...")
    
    # Create test file with error
    test_file = "test_error.py"
    test_content = '''def greet(name):
    message = f"Hello, {nam}!"  # Error: 'nam' instead of 'name'
    print(message)
    return message

result = greet("World")
print(result)
'''
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    try:
        assistant = AICodingAssistant()
        
        # Simulate error event
        error_event = ErrorEvent(
            timestamp=asyncio.get_event_loop().time(),
            error_type="runtime_error",
            language="python",
            message="NameError: name 'nam' is not defined",
            file_path=test_file,
            line_number=2,
            command="python test_error.py",
            working_directory=".",
            full_output="NameError: name 'nam' is not defined"
        )
        
        await assistant._handle_error_event(error_event)
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Coding Assistant')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.test:
        await test_mode()
    else:
        assistant = AICodingAssistant()
        await assistant.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n��� Goodbye!")
