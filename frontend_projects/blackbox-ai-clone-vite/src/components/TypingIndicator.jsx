import React from 'react';
import { Bot } from 'lucide-react';

const TypingIndicator = () => {
  return (
    <div className="flex space-x-3">
      <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
        <Bot size={16} />
      </div>
      <div className="flex-1">
        <div className="flex items-center space-x-2 mb-1">
          <span className="font-medium">BlackBox AI</span>
        </div>
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
