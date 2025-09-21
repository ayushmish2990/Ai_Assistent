import React from 'react';
import Header from './Header';
import MessageList from './MessageList';
import InputArea from './InputArea';
import { useChat } from '../hooks/useChat';

const ChatInterface = () => {
  const { messages, sendMessage, isTyping } = useChat();

  return (
    <div className="flex flex-col h-screen bg-black text-white">
      <Header />
      <MessageList messages={messages} isTyping={isTyping} />
      <InputArea onSendMessage={sendMessage} isDisabled={isTyping} />
    </div>
  );
};

export default ChatInterface;
