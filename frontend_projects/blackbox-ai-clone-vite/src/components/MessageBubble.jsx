import React from 'react';
import { User, Bot, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';
import { formatMessage } from '../utils/messageFormatter.jsx';

const MessageBubble = ({ message }) => {
  const isUser = message.type === 'user';

  return (
    <div className="flex space-x-3">
      <div className="flex-shrink-0">
        {isUser ? (
          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
            <User size={16} />
          </div>
        ) : (
          <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
            <Bot size={16} />
          </div>
        )}
      </div>
      <div className="flex-1">
        <div className="flex items-center space-x-2 mb-1">
          <span className="font-medium">
            {isUser ? 'You' : 'BlackBox AI'}
          </span>
          <span className="text-gray-500 text-sm">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
        <div className="text-gray-200 leading-relaxed">
          {formatMessage(message.content)}
        </div>
        {!isUser && (
          <div className="flex items-center space-x-2 mt-2">
            <button className="text-gray-500 hover:text-white p-1 transition-colors">
              <Copy size={16} />
            </button>
            <button className="text-gray-500 hover:text-green-500 p-1 transition-colors">
              <ThumbsUp size={16} />
            </button>
            <button className="text-gray-500 hover:text-red-500 p-1 transition-colors">
              <ThumbsDown size={16} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
