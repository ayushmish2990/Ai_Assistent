import React, { useState, useRef } from 'react';
import { Send, Code2, Image, Mic, Paperclip } from 'lucide-react';
import QuickActions from './QuickActions';

const InputArea = ({ onSendMessage, isDisabled }) => {
  const [inputValue, setInputValue] = useState('');
  const textareaRef = useRef(null);

  const handleSubmit = () => {
    if (!inputValue.trim() || isDisabled) return;
    onSendMessage(inputValue);
    setInputValue('');
    resetTextareaHeight();
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const resetTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '48px';
    }
  };

  const handleInput = (e) => {
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
  };

  return (
    <div className="p-4 border-t border-gray-800">
      <div className="relative">
        <div className="flex items-end space-x-2">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              onInput={handleInput}
              placeholder="Ask BlackBox AI anything..."
              className="w-full bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 pr-32 text-white placeholder-gray-400 resize-none focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-colors"
              rows="1"
              style={{ minHeight: '48px', maxHeight: '120px' }}
            />
            <div className="absolute right-2 bottom-2 flex items-center space-x-1">
              <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors">
                <Paperclip size={18} />
              </button>
              <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors">
                <Image size={18} />
              </button>
              <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors">
                <Code2 size={18} />
              </button>
              <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors">
                <Mic size={18} />
              </button>
            </div>
          </div>
          <button
            onClick={handleSubmit}
            disabled={!inputValue.trim() || isDisabled}
            className="p-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-xl transition-colors"
          >
            <Send size={20} />
          </button>
        </div>
      </div>
      <QuickActions onQuickAction={setInputValue} />
    </div>
  );
};

export default InputArea;
