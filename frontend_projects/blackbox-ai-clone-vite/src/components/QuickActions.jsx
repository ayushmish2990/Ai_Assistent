import React from 'react';

const QuickActions = ({ onQuickAction }) => {
  const quickActions = [
    { label: "Python Code", text: "Help me write a Python function" },
    { label: "Code Explanation", text: "Explain this code to me" },
    { label: "Debug Code", text: "Debug my JavaScript" },
    { label: "React Component", text: "Generate React component" }
  ];

  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {quickActions.map((action, index) => (
        <button
          key={index}
          onClick={() => onQuickAction(action.text)}
          className="px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded-full border border-gray-700 transition-colors"
        >
          {action.label}
        </button>
      ))}
    </div>
  );
};

export default QuickActions;
