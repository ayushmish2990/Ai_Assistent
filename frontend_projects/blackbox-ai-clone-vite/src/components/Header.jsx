import React from 'react';
import { Bot, MoreHorizontal } from 'lucide-react';

const Header = () => {
  return (
    <div className="flex items-center justify-between p-4 border-b border-gray-800">
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
          <Bot size={20} className="text-white" />
        </div>
        <h1 className="text-xl font-semibold">BlackBox AI</h1>
      </div>
      <button className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
        <MoreHorizontal size={20} />
      </button>
    </div>
  );
};

export default Header;
