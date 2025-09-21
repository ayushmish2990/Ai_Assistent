import { useState } from 'react';

export const useChat = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'assistant',
      content: "Hello! I'm BlackBox AI. I can help you with coding, answer questions, analyze images, and more. What would you like to work on today?",
      timestamp: new Date()
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);

  const sendMessage = async (content) => {
    const newMessage = {
      id: messages.length + 1,
      type: 'user',
      content,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, newMessage]);
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const responses = [
        "I'd be happy to help you with that! Could you provide more specific details about what you're looking for?",
        "Here's a solution for your request:\n\n```javascript\nfunction example() {\n  console.log('This is a sample response');\n  return 'BlackBox AI response';\n}\n```\n\nThis code demonstrates the functionality you requested.",
        "Based on your question, here are the key points to consider:\n\n1. First, analyze the requirements\n2. Break down the problem into smaller parts\n3. Implement step by step\n4. Test thoroughly\n\nWould you like me to elaborate on any of these points?",
      ];

      const aiResponse = {
        id: messages.length + 2,
        type: 'assistant',
        content: responses[Math.floor(Math.random() * responses.length)],
        timestamp: new Date()
      };


      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);
    }, 1000 + Math.random() * 2000);
  };

  return { messages, sendMessage, isTyping };
};
