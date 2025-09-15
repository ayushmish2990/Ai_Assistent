"""AI Service for advanced code intelligence and generation.

This service integrates with OpenAI GPT models to provide Cursor-like AI capabilities
including code completion, analysis, generation, and intelligent suggestions.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from pydantic import BaseModel
from app.core.config import settings

logger = logging.getLogger(__name__)

class CodeContext(BaseModel):
    """Context information for code analysis and generation."""
    file_path: Optional[str] = None
    language: Optional[str] = None
    content: Optional[str] = None
    imports: List[str] = []
    cursor_position: Optional[int] = None
    selected_text: Optional[str] = None
    surrounding_code: Optional[str] = None
    project_context: Optional[Dict[str, Any]] = None

class AIResponse(BaseModel):
    """Response from AI service."""
    content: str
    suggestions: List[str] = []
    confidence: float = 0.0
    reasoning: Optional[str] = None
    code_blocks: List[Dict[str, str]] = []

class AIService:
    """Advanced AI service for code intelligence."""
    
    def __init__(self):
        api_key = settings.OPENAI_API_KEY
        if api_key and api_key != "your_openai_api_key_here":
            try:
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            self.client = None
            logger.warning("OpenAI API key not configured. AI features will be disabled.")
        self.model = settings.OPENAI_MODEL
        
    async def analyze_code(self, code: str, context: CodeContext = None) -> AIResponse:
        """Analyze code for issues, suggestions, and improvements."""
        if self.client is None:
            return AIResponse(
                content="AI analysis is not available. Please configure OPENAI_API_KEY in your .env file.",
                confidence=0.0,
                suggestions=["Configure OpenAI API key to enable AI features"]
            )
            
        try:
            system_prompt = self._get_analysis_system_prompt()
            user_prompt = self._build_analysis_prompt(code, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            return self._parse_analysis_response(content)
            
        except Exception as e:
            logger.error(f"Error in code analysis: {e}")
            return AIResponse(
                content=f"Error analyzing code: {str(e)}",
                confidence=0.0
            )
    
    async def generate_code(self, prompt: str, context: CodeContext = None) -> AIResponse:
        """Generate code based on natural language description."""
        if self.client is None:
            return AIResponse(
                content="Code generation is not available. Please configure OPENAI_API_KEY in your .env file.",
                confidence=0.0,
                suggestions=["Configure OpenAI API key to enable AI features"]
            )
            
        try:
            system_prompt = self._get_generation_system_prompt()
            user_prompt = self._build_generation_prompt(prompt, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            return self._parse_generation_response(content)
            
        except Exception as e:
            logger.error(f"Error in code generation: {e}")
            error_msg = str(e)
            if "insufficient_quota" in error_msg or "429" in error_msg:
                return AIResponse(
                    content="AI services are temporarily unavailable due to quota limits. Please check your OpenAI billing and usage limits.",
                    confidence=0.0,
                    reasoning="OpenAI API quota exceeded"
                )
            else:
                return AIResponse(
                    content=f"Error generating code: {error_msg}",
                    confidence=0.0
                )
    
    async def suggest_completion(self, code: str, cursor_pos: int, context: CodeContext = None) -> AIResponse:
        """Suggest code completion at cursor position (Cursor-like autocomplete)."""
        if self.client is None:
            return AIResponse(
                content="Code completion is not available. Please configure OPENAI_API_KEY in your .env file.",
                confidence=0.0,
                suggestions=["Configure OpenAI API key to enable AI features"]
            )
            
        try:
            # Extract code before and after cursor
            before_cursor = code[:cursor_pos]
            after_cursor = code[cursor_pos:]
            
            system_prompt = self._get_completion_system_prompt()
            user_prompt = self._build_completion_prompt(before_cursor, after_cursor, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            return self._parse_completion_response(content)
            
        except Exception as e:
            logger.error(f"Error in code completion: {e}")
            error_msg = str(e)
            if "insufficient_quota" in error_msg or "429" in error_msg:
                return AIResponse(
                    content="AI completion services are temporarily unavailable due to quota limits. Please check your OpenAI billing and usage limits.",
                    confidence=0.0,
                    reasoning="OpenAI API quota exceeded"
                )
            else:
                return AIResponse(
                    content="",
                    confidence=0.0
                )
    
    async def chat_with_context(self, message: str, conversation_history: List[Dict], 
                               codebase_context: Optional[Dict] = None) -> AIResponse:
        """Enhanced chat with codebase context awareness."""
        if self.client is None:
            return AIResponse(
                content="AI chat is not available. Please configure OPENAI_API_KEY in your .env file to enable AI features.",
                confidence=0.0
            )
            
        try:
            system_prompt = self._get_chat_system_prompt(codebase_context)
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history[-10:])  # Last 10 messages for context
            messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            return AIResponse(
                content=content,
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_msg = str(e)
            if "insufficient_quota" in error_msg or "429" in error_msg:
                return AIResponse(
                    content="AI chat services are temporarily unavailable due to quota limits. Please check your OpenAI billing and usage limits at https://platform.openai.com/account/billing.",
                    confidence=0.0,
                    reasoning="OpenAI API quota exceeded"
                )
            else:
                return AIResponse(
                    content=f"I apologize, but I encountered an error: {error_msg}",
                    confidence=0.0
                )
    
    def _get_analysis_system_prompt(self) -> str:
        return """
You are an expert code analyzer with deep understanding of multiple programming languages.
Analyze the provided code for:
1. Potential bugs and issues
2. Performance improvements
3. Code quality and best practices
4. Security vulnerabilities
5. Refactoring suggestions

Provide clear, actionable feedback with specific line references when possible.
"""
    
    def _get_generation_system_prompt(self) -> str:
        return """
You are an expert software developer capable of generating high-quality, production-ready code.
Generate code that is:
1. Functional and correct
2. Well-structured and readable
3. Following best practices
4. Properly documented
5. Secure and efficient

Always include necessary imports and handle edge cases.
"""
    
    def _get_completion_system_prompt(self) -> str:
        return """
You are an intelligent code completion assistant like Cursor AI.
Predict the most likely code continuation at the cursor position.
Consider:
1. Code context and patterns
2. Variable names and types
3. Function signatures
4. Language-specific idioms
5. Project structure

Provide only the completion text, no explanations.
"""
    
    def _get_chat_system_prompt(self, codebase_context: Optional[Dict] = None) -> str:
        base_prompt = """
You are an AI coding assistant with deep understanding of software development.
Help users with coding questions, debugging, architecture decisions, and development tasks.
"""
        
        if codebase_context:
            base_prompt += f"\n\nCodebase context: {codebase_context}"
        
        return base_prompt
    
    def _build_analysis_prompt(self, code: str, context: CodeContext = None) -> str:
        prompt = f"Analyze this code:\n\n```\n{code}\n```\n\n"
        
        if context:
            if context.language:
                prompt += f"Language: {context.language}\n"
            if context.file_path:
                prompt += f"File: {context.file_path}\n"
        
        return prompt
    
    def _build_generation_prompt(self, prompt: str, context: CodeContext = None) -> str:
        full_prompt = f"Generate code for: {prompt}\n\n"
        
        if context:
            if context.language:
                full_prompt += f"Target language: {context.language}\n"
            if context.surrounding_code:
                full_prompt += f"Context code:\n```\n{context.surrounding_code}\n```\n\n"
        
        return full_prompt
    
    def _build_completion_prompt(self, before: str, after: str, context: CodeContext = None) -> str:
        prompt = f"Complete the code at <CURSOR>:\n\n```\n{before}<CURSOR>{after}\n```\n\n"
        
        if context and context.language:
            prompt += f"Language: {context.language}\n"
        
        return prompt
    
    def _parse_analysis_response(self, content: str) -> AIResponse:
        # Extract suggestions and code blocks from response
        lines = content.split('\n')
        suggestions = []
        code_blocks = []
        
        current_suggestion = ""
        in_code_block = False
        current_code = ""
        code_language = ""
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    code_blocks.append({
                        "language": code_language,
                        "code": current_code.strip()
                    })
                    current_code = ""
                    in_code_block = False
                else:
                    code_language = line.strip()[3:].strip()
                    in_code_block = True
            elif in_code_block:
                current_code += line + "\n"
            elif line.strip().startswith(('-', '*', '•')):
                if current_suggestion:
                    suggestions.append(current_suggestion.strip())
                current_suggestion = line.strip()[1:].strip()
            elif current_suggestion:
                current_suggestion += " " + line.strip()
        
        if current_suggestion:
            suggestions.append(current_suggestion.strip())
        
        return AIResponse(
            content=content,
            suggestions=suggestions,
            code_blocks=code_blocks,
            confidence=0.8
        )
    
    def _parse_generation_response(self, content: str) -> AIResponse:
        return self._parse_analysis_response(content)
    
    def _parse_completion_response(self, content: str) -> AIResponse:
        # For completion, we want just the suggested text
        # Remove any markdown formatting
        clean_content = content.strip()
        if clean_content.startswith('```') and clean_content.endswith('```'):
            lines = clean_content.split('\n')
            clean_content = '\n'.join(lines[1:-1])
        
        return AIResponse(
            content=clean_content,
            confidence=0.9
        )