import { useState } from 'react';
import { 
  Menu, 
  Plus, 
  MessageSquare, 
  Settings, 
  HelpCircle,
  X
} from 'lucide-react';

const Sidebar = ({ isOpen, onClose }) => {
  const [activeItem, setActiveItem] = useState('new-chat');

  const menuItems = [
    { id: 'new-chat', icon: Plus, label: 'New chat' },
    { id: 'history', icon: MessageSquare, label: 'Chat history' },
    { id: 'settings', icon: Settings, label: 'Settings' },
    { id: 'help', icon: HelpCircle, label: 'Help & FAQ' },
  ];

  return (
    <>
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}
      <div 
        className={`fixed top-0 left-0 h-full w-64 bg-background border-r z-50 transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } lg:translate-x-0`}
      >
        <div className="h-full flex flex-col">
          <div className="p-4 border-b flex items-center justify-between">
            <h2 className="text-lg font-semibold">Blackbox AI</h2>
            <button 
              onClick={onClose}
              className="lg:hidden p-1 rounded-md hover:bg-accent"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
          
          <div className="flex-1 overflow-y-auto p-2">
            <button className="w-full flex items-center gap-3 p-3 rounded-md hover:bg-accent text-sm font-medium">
              <Plus className="h-4 w-4" />
              New chat
            </button>
            
            <div className="mt-4">
              <h3 className="px-3 text-xs font-semibold text-muted-foreground mb-2">
                Recent
              </h3>
              <div className="space-y-1">
                <button className="w-full text-left px-3 py-2 text-sm rounded-md hover:bg-accent truncate">
                  How to use the API
                </button>
                <button className="w-full text-left px-3 py-2 text-sm rounded-md hover:bg-accent truncate">
                  Python code examples
                </button>
              </div>
            </div>
          </div>
          
          <div className="p-4 border-t">
            <button className="w-full flex items-center gap-3 p-3 rounded-md hover:bg-accent text-sm font-medium">
              <Settings className="h-4 w-4" />
              Settings
            </button>
            <button className="w-full flex items-center gap-3 p-3 rounded-md hover:bg-accent text-sm font-medium">
              <HelpCircle className="h-4 w-4" />
              Help & FAQ
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;
