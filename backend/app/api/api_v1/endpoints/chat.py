from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import random
from app.services.ai_service import AIService, CodeContext

router = APIRouter()
ai_service = AIService()

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """
    Handle general AI chat conversations with enhanced AI capabilities.
    """
    try:
        user_message = request.message.strip()
        
        if not user_message:
            return ChatResponse(
                response="I'm here to help! Please ask me a question or tell me what you'd like to work on."
            )
        
        # Handle conversation history (convert None to empty list)
        conversation_history = request.conversation_history or []
        
        # Convert to format expected by AI service
        formatted_history = [
            {"role": msg.role, "content": msg.content} 
            for msg in conversation_history
        ]
        
        # Use AI service for enhanced responses
        ai_response = await ai_service.chat_with_context(
            message=user_message,
            conversation_history=formatted_history
        )
        
        return ChatResponse(
            response=ai_response.content,
            conversation_id="chat_session_1"  # Simple session ID for now
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

def generate_ai_response(message: str) -> str:
    """
    Generate AI responses based on user input.
    This is a simple implementation - in production, you'd integrate with a real AI model.
    """
    message_lower = message.lower()
    
    # Greeting responses
    if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        greetings = [
            "Hello! I'm BlackBox AI, your coding assistant. How can I help you today?",
            "Hi there! I'm here to help with coding, debugging, and answering your questions.",
            "Hey! What would you like to work on today? I can help with code, explanations, or general questions."
        ]
        return random.choice(greetings)
    
    # Coding-related questions
    if any(keyword in message_lower for keyword in ['code', 'programming', 'function', 'algorithm', 'debug', 'error']):
        return "I'd be happy to help with your coding question! Could you share the specific code you're working with, or tell me more about what you're trying to accomplish?"
    
    # Help requests
    if any(keyword in message_lower for keyword in ['help', 'assist', 'support']):
        return "I'm here to help! I can assist with:\n\n• Code analysis and debugging\n• Programming questions and explanations\n• Algorithm design and optimization\n• Best practices and code reviews\n• General programming concepts\n\nWhat would you like to work on?"
    
    # Questions about capabilities
    if any(keyword in message_lower for keyword in ['what can you do', 'capabilities', 'features']):
        return "I'm BlackBox AI, and I can help you with:\n\n🔍 **Code Analysis**: Review your code for bugs and improvements\n💡 **Programming Help**: Answer questions about various programming languages\n🛠️ **Debugging**: Help identify and fix issues in your code\n📚 **Learning**: Explain programming concepts and best practices\n⚡ **Optimization**: Suggest ways to improve code performance\n\nJust share your code or ask me anything programming-related!"
    
    # Thank you responses
    if any(keyword in message_lower for keyword in ['thank', 'thanks']):
        return "You're welcome! Feel free to ask if you need any more help with your coding projects."
    
    # Default responses for general questions
    general_responses = [
        "That's an interesting question! Could you provide more details so I can give you a better answer?",
        "I'd be happy to help! Can you tell me more about what you're working on?",
        "I'm here to assist with programming and coding questions. What specific challenge are you facing?",
        "Could you elaborate on that? I want to make sure I give you the most helpful response."
    ]
    
    return random.choice(general_responses)