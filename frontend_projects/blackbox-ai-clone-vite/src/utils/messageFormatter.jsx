import React from 'react';
import { Copy } from 'lucide-react';

export const formatMessage = (content) => {
  const parts = content.split('```');
  
  return parts.map((part, index) => {
    if (index % 2 === 1) {
      const lines = part.split('\n');
      const language = lines[0] || 'code';
      const code = lines.slice(1).join('\n');
      
      return (
        <div key={index} className="bg-gray-900 rounded-lg p-4 my-3 overflow-x-auto">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400 text-sm">{language}</span>
            <button className="text-gray-400 hover:text-white transition-colors">
              <Copy size={16} />
            </button>
          </div>
          <pre className="text-green-400 text-sm">
            <code>{code}</code>
          </pre>
        </div>
      );
    }
    
    return part.split('\n').map((line, lineIndex) => (
      <div key={`${index}-${lineIndex}`}>
        {line}
        {lineIndex < part.split('\n').length - 1 && <br />}
      </div>
    ));
  });
};
