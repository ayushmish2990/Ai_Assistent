#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blackbox - Main Application
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitors.terminal_monitor import TerminalMonitor, ErrorEvent
from parsers.error_parser import ErrorParser
from processors.factory import ProcessorFactory
from fixers.code_fixer import CodeFixer, CodeApplier

class BlackboxAssistant:
    """Main application that coordinates all components"""
    
    def __init__(self):
        self.terminal_monitor = TerminalMonitor()
        self.processor_factory = ProcessorFactory()
        self.code_fixer = CodeFixer()  # For show_diff
        self.code_applier = CodeApplier()
        
        self.stats = {
            'errors_detected': 0,
            'fixes_generated': 0,
            'fixes_applied': 0
        }
        
        # Register error handler
        self.terminal_monitor.register_callback(self._handle_error_event)

        # Setup logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.event_log_path = self.log_dir / "events.jsonl"
        
        logger.info("Blackbox initialized")

    async def _get_fix_suggestion(self, error_event: ErrorEvent):
        """Analyzes an error event and returns a fix suggestion."""
        processor = self.processor_factory.get_processor(error_event.language)
        if not processor:
            logger.warning(f"No processor available for language: {error_event.language}")
            return None, None

        # Analyze the error
        error_analysis = processor.analyze_error(error_event)
        if not error_analysis:
            logger.warning("Could not analyze error")
            return None, None

        # Generate fix suggestion
        fix_suggestion = await processor.generate_fix(error_analysis)
        return error_analysis, fix_suggestion

    async def _handle_error_event(self, error_event: ErrorEvent):
        """Handle detected error events"""
        try:
            self.stats['errors_detected'] += 1
            
            print(f"\n🕯️ Error detected: {error_event.error_type} in {error_event.language}")
            print(f"🕯️ File: {error_event.file_path}:{error_event.line_number}")
            print(f"🕯️ Message: {error_event.message}")

            error_analysis, fix_suggestion = await self._get_fix_suggestion(error_event)
            
            if not error_analysis or not fix_suggestion:
                print("❌ Could not generate fix suggestion")
                return
            
            print(f"🔍 Analysis: {error_analysis.error_type} (confidence: {error_analysis.confidence:.2f})")
            
            self.stats['fixes_generated'] += 1
            print(f"💡 Fix suggested: {fix_suggestion.description}")
            print(f"💡 Confidence: {fix_suggestion.confidence:.2f}")
            
            # Show diff
            print("\n💡 Proposed changes:")
            print(self.code_fixer.show_diff(fix_suggestion))
            
            # Ask for confirmation
            response = input("\n❓ Apply this fix? [y/n]: ").lower().strip()
            
            action = "skipped"
            success = None
            if response in ['y', 'yes']:
                action = "applied"
                result = await self.code_applier.apply_fix(fix_suggestion)
                if result.success:
                    self.stats['fixes_applied'] += 1
                    print(f"✅ {result.message}")
                    success = True
                else:
                    print(f"❌ {result.message}")
                    success = False
            else:
                print("⏭️  Fix skipped")

            self._log_event("error_handled", {
                "error": error_event,
                "analysis": error_analysis,
                "suggestion": fix_suggestion,
                "user_action": action,
                "fix_applied_success": success
            })
                
        except Exception as e:
            logger.error(f"Error handling error event: {e}")
            self._log_event("error_handling_failed", {"error": str(e)})

    def _log_event(self, event_type: str, data: dict):
        """Logs an event to the events.jsonl file."""
        def default_serializer(o):
            if hasattr(o, '_asdict'):
                return o._asdict()
            if hasattr(o, '__dict__'):
                return o.__dict__
            return str(o)

        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "data": data
            }
            with open(self.event_log_path, 'a') as f:
                f.write(json.dumps(log_entry, default=default_serializer) + '\n')
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    async def start(self):
        """Start the Blackbox assistant"""
        print("\n" + "="*60)
        print("⬛ BLACKBOX STARTED")
        print("="*60)
        print("🤖 Monitoring for programming errors...")
        print("💡 Ready to suggest fixes")
        print("⚠️  Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        try:
            await self.terminal_monitor.start_monitoring()
        except KeyboardInterrupt:
            await self.stop()
    
    async def stop(self):
        """Stop the Blackbox assistant"""
        print("\n🛑 Stopping Blackbox...")
        self.terminal_monitor.stop_monitoring()
        
        print(f"\n📊 Session Statistics:")
        print(f"🔍 Errors detected: {self.stats['errors_detected']}")
        print(f"💡 Fixes generated: {self.stats['fixes_generated']}")
        print(f"✅ Fixes applied: {self.stats['fixes_applied']}")
        
        print("\n👋 Goodbye!")

async def test_mode():
    """Test mode with simulated errors"""
    print("🧪 Running in test mode...")
    
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
        assistant = BlackboxAssistant()
        
        # Simulate error event
        error_event = ErrorEvent(
            timestamp=asyncio.get_event_loop().time(),
            error_type="NameError",
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

async def test_js_mode():
    """Test mode with simulated JavaScript error"""
    print("🧪 Running in JS test mode...")
    
    # Create test file with error
    test_file = "test_error.js"
    test_content = '''function greet(name) {
    console.log(mesage); // Error: 'mesage' instead of 'message'
}

greet("World");
'''
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    try:
        assistant = BlackboxAssistant()
        
        # Simulate error event
        error_event = ErrorEvent(
            timestamp=asyncio.get_event_loop().time(),
            error_type="ReferenceError",
            language="javascript",
            message="ReferenceError: mesage is not defined",
            file_path=test_file,
            line_number=2,
            command="node test_error.js",
            working_directory=".",
            full_output="ReferenceError: mesage is not defined"
        )
        
        await assistant._handle_error_event(error_event)
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

async def analyze_file(file_path: str):
    """Analyzes a single file for errors and suggests a fix as JSON."""
    assistant = BlackboxAssistant()
    language = "python" if file_path.endswith(".py") else "javascript"
    command = [sys.executable, file_path] if language == "python" else ["node", file_path]

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        error_output = stderr.decode()
        if not error_output:
            print(json.dumps({"error": "No error detected in file."}))
            return

        parser = ErrorParser()
        error_event = parser.parse(error_output)

        if not error_event:
            print(json.dumps({"error": "Could not parse error output.", "details": error_output}))
            return
        
        error_event = error_event._replace(file_path=file_path, language=language)

        error_analysis, fix_suggestion = await assistant._get_fix_suggestion(error_event)

        if not error_analysis or not fix_suggestion:
            print(json.dumps({"error": "Could not generate fix suggestion."}))
            return
        
        def default_serializer(o):
            if hasattr(o, '_asdict'):
                return o._asdict()
            if hasattr(o, '__dict__'):
                return o.__dict__
            return str(o)

        result = {
            "analysis": error_analysis,
            "suggestion": fix_suggestion,
            "diff": assistant.code_fixer.show_diff(fix_suggestion)
        }
        print(json.dumps(result, default=default_serializer, indent=2))

    except Exception as e:
        print(json.dumps({"error": f"Failed to analyze file: {str(e)}"}))


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Blackbox AI Coding Assistant')
    parser.add_argument('--test', action='store_true', help='Run in test mode for Python')
    parser.add_argument('--test-js', action='store_true', help='Run in test mode for JavaScript')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--analyze-file', type=str, help='Analyze a single file and output JSON')
    
    args = parser.parse_args()
    
    if args.analyze_file:
        await analyze_file(args.analyze_file)
    elif args.test:
        await test_mode()
    elif args.test_js:
        await test_js_mode()
    else:
        assistant = BlackboxAssistant()
        await assistant.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
