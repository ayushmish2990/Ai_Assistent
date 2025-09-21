import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User } from 'lucide-react';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    { id: 1, text: 'Hello! How can I help you today?', sender: 'bot' }
  ]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage = { id: messages.length + 1, text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');

    // Simulate bot response
    setTimeout(() => {
      const botMessage = { 
        id: messages.length + 2, 
        text: `I received: "${input}"`,
        sender: 'bot' 
      };
      setMessages(prev => [...prev, botMessage]);
    }, 1000);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div 
            key={message.id} 
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div 
              className={`flex items-start gap-2 max-w-[80%] p-3 rounded-lg ${
                message.sender === 'user' 
                  ? 'bg-primary text-primary-foreground' 
                  : 'bg-muted'
              }`}
            >
              {message.sender === 'bot' ? (
                <Bot className="h-5 w-5 mt-1 text-muted-foreground" />
              ) : (
                <User className="h-5 w-5 mt-1 text-primary-foreground" />
              )}
              <p className="text-sm">{message.text}</p>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="border-t p-4">
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 rounded-lg border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          />
          <button
            type="submit"
            className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2"
            disabled={!input.trim()}
          >
            <Send className="h-4 w-4 mr-2" />
            Send
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;
