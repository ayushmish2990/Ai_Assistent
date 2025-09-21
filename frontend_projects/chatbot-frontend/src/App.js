import React, { useState, useRef, useEffect } from 'react';
import { Send, Code2, Image, Mic, Paperclip, MoreHorizontal, User, Bot, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';
import './App.css';
import api from './services/api';

const App = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'assistant',
      content: "Hello! I'm BlackBox AI. I can help you with coding, answer questions, analyze images, and more. What would you like to work on today?",
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [codeInput, setCodeInput] = useState('');
  const [language, setLanguage] = useState('python');
  const [completions, setCompletions] = useState([]);
  const [showCompletions, setShowCompletions] = useState(false);
  const [codebaseStatus, setCodebaseStatus] = useState(null);
  const [codeErrors, setCodeErrors] = useState([]);
  const [showErrors, setShowErrors] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const codeTextareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const isCodeInput = (text) => {
    // Check for common code patterns
    const codePatterns = [
      /\b(function|class|def|public|private|protected|static)\b/i,
      /\b(import|include|require|from)\s+[\w.]+/i,
      /\b(if|else|for|while|switch|case)\s*\(/i,
      /[{}();]\s*$/m,
      /^\s*(#include|import|from|package)/m,
      /\b(console\.log|print|printf|System\.out)\s*\(/i,
      /\b(var|let|const|int|string|boolean|float|double)\b/i,
      /[=+\-*/%]\s*[\w[\]]+\s*[;,]/,
      /\breturn\s+[\w[\]"']+/i
    ];
    
    return codePatterns.some(pattern => pattern.test(text)) || 
           text.includes('//') || 
           text.includes('/*') || 
           text.includes('*/') ||
           (text.includes('{') && text.includes('}'));
  };

  const handleCodeCompletion = async (code, cursorPosition) => {
    try {
      const response = await api.completeCode(code, cursorPosition, language);
      setCompletions(response.completions || []);
      setShowCompletions(response.completions && response.completions.length > 0);
    } catch (error) {
      console.error('Code completion failed:', error);
    }
  };

  const handleCodeAnalysis = async () => {
    if (!codeInput.trim()) return;
    
    setIsTyping(true);
    try {
      const response = await api.analyzeCode(codeInput, language);
      const botMessage = {
        id: Date.now(),
        type: 'assistant',
        content: `Code Analysis Results:\n\nLanguage: ${response.language}\n\nSuggestions:\n${response.suggestions.join('\n')}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Code analysis failed:', error);
    } finally {
      setIsTyping(false);
    }
  };

  const handleCodeGeneration = async (prompt) => {
    setIsTyping(true);
    try {
      const response = await api.generateCode(prompt, language);
      setCodeInput(response.generated_code);
      const botMessage = {
        id: Date.now(),
        type: 'assistant',
        content: `Generated ${language} code:\n\n\`\`\`${language}\n${response.generated_code}\n\`\`\`\n\n${response.explanation || ''}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Code generation failed:', error);
    } finally {
      setIsTyping(false);
    }
  };

  const handleErrorAnalysis = async () => {
    if (!codeInput.trim()) return;
    
    setIsAnalyzing(true);
    try {
      const result = await api.analyzeCodeErrors(codeInput, language);
      setCodeErrors(result.errors);
      setShowErrors(true);
    } catch (error) {
      console.error('Error analyzing code errors:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAutoFix = async () => {
    if (!codeInput.trim()) return;
    
    try {
      const result = await api.autoFixCode(codeInput, language);
      setCodeInput(result.fixed_code);
      // Re-analyze after fixing
      const analysisResult = await api.analyzeCodeErrors(result.fixed_code, language);
      setCodeErrors(analysisResult.errors);
    } catch (error) {
      console.error('Error auto-fixing code:', error);
    }
  };

  useEffect(() => {
    const fetchCodebaseStatus = async () => {
      try {
        const status = await api.getCodebaseStatus();
        setCodebaseStatus(status);
      } catch (error) {
        console.error('Failed to fetch codebase status:', error);
      }
    };
    fetchCodebaseStatus();
  }, []);

  const handleSubmit = async () => {
    if (!inputValue.trim()) return;

    const newMessage = {
      id: messages.length + 1,
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, newMessage]);
    const currentInput = inputValue;
    setInputValue('');
    setIsTyping(true);

    try {
      let responseContent;
      
      if (isCodeInput(currentInput)) {
        // Route to code analysis endpoint
        const analysisResult = await api.analyzeCode(currentInput);
        
        if (analysisResult.suggestions && analysisResult.suggestions.length > 0) {
          // Handle code generation responses
          responseContent = `${analysisResult.details || ''}\n\n${analysisResult.suggestions.join('\n\n')}`;
        } else if (analysisResult.error && analysisResult.error !== 'No error detected in file.') {
          // Handle code analysis with issues
          responseContent = `I found some issues in your code:\n\n${analysisResult.error}\n\n${analysisResult.details || ''}`;
        } else {
          // Handle successful code analysis or general responses
          responseContent = analysisResult.details || "I've analyzed your code and it looks good! No errors were detected.";
        }
      } else {
        // Route to chat endpoint for general conversation
        const result = await api.sendChatMessage(currentInput, messages.filter(msg => msg.sender === 'bot').slice(-5));
        responseContent = result.response;
      }
      
      const aiResponse = {
        id: messages.length + 2,
        type: 'assistant',
        content: responseContent,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, aiResponse]);
    } catch (error) {
      console.error('Error processing request:', error);
      
      const errorResponse = {
        id: messages.length + 2,
        type: 'assistant',
        content: "I'm sorry, I encountered an error while processing your request. Please try again later.",
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const formatMessage = (content) => {
    // Simple code block formatting
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
              <button className="text-gray-400 hover:text-white">
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="bg-black/20 backdrop-blur-sm border-b border-white/10">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Code2 className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold text-white">AI Coding Assistant</h1>
              {codebaseStatus && (
                <div className="text-sm text-gray-400">
                  {codebaseStatus.total_files} files indexed
                </div>
              )}
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex bg-black/30 rounded-lg p-1">
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === 'chat'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Chat
                </button>
                <button
                  onClick={() => setActiveTab('code')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === 'code'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Code Editor
                </button>
              </div>
              <button className="p-2 text-gray-400 hover:text-white transition-colors">
                <MoreHorizontal className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-6">
        {activeTab === 'chat' ? (
          <div className="bg-black/20 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
            {/* Messages Area */}
            <div className="h-96 overflow-y-auto p-6 space-y-4">
              {messages.length === 0 ? (
                <div className="text-center text-gray-400 mt-20">
                  <Code2 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p className="text-lg mb-2">Welcome to AI Coding Assistant</p>
                  <p className="text-sm">Ask me anything about code, or paste code for analysis!</p>
                </div>
              ) : (
                messages.map((message) => (
                  <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`flex items-start space-x-3 max-w-3xl ${
                      message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                    }`}>
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                        message.type === 'user' 
                          ? 'bg-blue-600' 
                          : 'bg-gradient-to-r from-purple-500 to-pink-500'
                      }`}>
                        {message.type === 'user' ? (
                          <User className="w-4 h-4 text-white" />
                        ) : (
                          <Bot className="w-4 h-4 text-white" />
                        )}
                      </div>
                      <div className={`rounded-2xl px-4 py-3 ${
                        message.type === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-white/10 text-white border border-white/20'
                      }`}>
                        <div className="whitespace-pre-wrap text-sm leading-relaxed">
                          {formatMessage(message.content)}
                        </div>
                        <div className="text-xs opacity-70 mt-2">
                          {message.timestamp.toLocaleTimeString()}
                        </div>
                        {message.type === 'assistant' && (
                          <div className="flex items-center space-x-2 mt-3 pt-2 border-t border-white/10">
                            <button className="p-1 text-gray-400 hover:text-white transition-colors">
                              <Copy className="w-4 h-4" />
                            </button>
                            <button className="p-1 text-gray-400 hover:text-green-400 transition-colors">
                              <ThumbsUp className="w-4 h-4" />
                            </button>
                            <button className="p-1 text-gray-400 hover:text-red-400 transition-colors">
                              <ThumbsDown className="w-4 h-4" />
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
              {isTyping && (
                <div className="flex justify-start">
                  <div className="flex items-start space-x-3 max-w-3xl">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center flex-shrink-0">
                      <Bot className="w-4 h-4 text-white" />
                    </div>
                    <div className="bg-white/10 rounded-2xl px-4 py-3 border border-white/20">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="border-t border-white/10 p-4">
              <div className="flex items-end space-x-3">
                <div className="flex-1 relative">
                  <textarea
                    ref={textareaRef}
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask me anything about code, or paste your code here..."
                    className="w-full bg-white/5 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows="3"
                    disabled={isTyping}
                  />
                </div>
                <div className="flex flex-col space-y-2">
                  <button className="p-2 text-gray-400 hover:text-white transition-colors">
                    <Paperclip className="w-5 h-5" />
                  </button>
                  <button className="p-2 text-gray-400 hover:text-white transition-colors">
                    <Image className="w-5 h-5" />
                  </button>
                  <button className="p-2 text-gray-400 hover:text-white transition-colors">
                    <Mic className="w-5 h-5" />
                  </button>
                </div>
                <button
                  onClick={handleSubmit}
                  disabled={!inputValue.trim() || isTyping}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white p-3 rounded-xl transition-colors flex items-center justify-center"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Code Editor */}
            <div className="bg-black/20 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
              <div className="border-b border-white/10 p-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-white font-medium">Code Editor</h3>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="bg-white/10 border border-white/20 rounded-lg px-3 py-1 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="python">Python</option>
                    <option value="javascript">JavaScript</option>
                    <option value="typescript">TypeScript</option>
                    <option value="java">Java</option>
                    <option value="cpp">C++</option>
                    <option value="go">Go</option>
                    <option value="rust">Rust</option>
                  </select>
                </div>
              </div>
              <div className="relative">
                <textarea
                  ref={codeTextareaRef}
                  value={codeInput}
                  onChange={(e) => {
                    setCodeInput(e.target.value);
                    const cursorPos = e.target.selectionStart;
                    if (e.target.value.length > 10) {
                      handleCodeCompletion(e.target.value, cursorPos);
                    }
                  }}
                  placeholder={`Enter your ${language} code here...`}
                  className="w-full h-96 bg-transparent text-white font-mono text-sm p-4 resize-none focus:outline-none"
                  spellCheck={false}
                />
                {showCompletions && completions.length > 0 && (
                  <div className="absolute top-4 right-4 bg-black/80 border border-white/20 rounded-lg p-2 max-w-xs">
                    <div className="text-xs text-gray-400 mb-2">Suggestions:</div>
                    {completions.slice(0, 3).map((completion, index) => (
                      <div
                        key={index}
                        className="text-sm text-white p-2 hover:bg-white/10 rounded cursor-pointer"
                        onClick={() => {
                          setCodeInput(prev => prev + completion);
                          setShowCompletions(false);
                        }}
                      >
                        {completion.substring(0, 50)}...
                      </div>
                    ))}
                  </div>
                )}
                {/* Error Detection Panel */}
                {showErrors && (
                  <div className="absolute top-4 left-4 right-4 bg-black/90 border border-white/20 rounded-lg p-4 max-h-60 overflow-y-auto">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-lg font-semibold text-white flex items-center">
                        <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
                        Code Analysis Results
                      </h3>
                      <button
                        onClick={() => setShowErrors(false)}
                        className="text-gray-400 hover:text-white"
                      >
                        ×
                      </button>
                    </div>
                    
                    {codeErrors.length === 0 ? (
                      <div className="text-green-400 text-sm">
                        ✓ No issues found in your code!
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {codeErrors.map((error, index) => (
                          <div
                            key={index}
                            className={`p-3 rounded border-l-4 ${
                              error.severity === 'error'
                                ? 'bg-red-900/20 border-red-500'
                                : error.severity === 'warning'
                                ? 'bg-yellow-900/20 border-yellow-500'
                                : error.severity === 'info'
                                ? 'bg-blue-900/20 border-blue-500'
                                : 'bg-gray-900/20 border-gray-500'
                            }`}
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <div className="flex items-center space-x-2 mb-1">
                                  <span className={`text-xs px-2 py-1 rounded ${
                                    error.severity === 'error'
                                      ? 'bg-red-600 text-white'
                                      : error.severity === 'warning'
                                      ? 'bg-yellow-600 text-white'
                                      : error.severity === 'info'
                                      ? 'bg-blue-600 text-white'
                                      : 'bg-gray-600 text-white'
                                  }`}>
                                    {error.severity.toUpperCase()}
                                  </span>
                                  <span className="text-gray-400 text-sm">
                                    Line {error.line}, Column {error.column}
                                  </span>
                                </div>
                                <div className="text-white text-sm mb-1">
                                  {error.message}
                                </div>
                                {error.suggestion && (
                                  <div className="text-gray-300 text-xs">
                                    💡 {error.suggestion}
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
              <div className="border-t border-white/10 p-4">
                <div className="flex space-x-2">
                  <button
                    onClick={handleCodeAnalysis}
                    disabled={!codeInput.trim() || isTyping}
                    className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg text-sm transition-colors"
                  >
                    Analyze Code
                  </button>
                  <button
                    onClick={() => handleCodeGeneration(inputValue)}
                    disabled={!inputValue.trim() || isTyping}
                    className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg text-sm transition-colors"
                  >
                    Generate from Chat
                  </button>
                  <button
                    onClick={handleErrorAnalysis}
                    disabled={isAnalyzing}
                    className="bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg text-sm transition-colors"
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Check Errors'}
                  </button>
                  <button
                    onClick={handleAutoFix}
                    className="bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg text-sm transition-colors"
                  >
                    Auto Fix
                  </button>
                </div>
              </div>
            </div>

            {/* Chat Panel */}
            <div className="bg-black/20 backdrop-blur-sm rounded-2xl border border-white/10 overflow-hidden">
              <div className="border-b border-white/10 p-4">
                <h3 className="text-white font-medium">AI Assistant</h3>
              </div>
              <div className="h-80 overflow-y-auto p-4 space-y-3">
                {messages.slice(-5).map((message) => (
                  <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`rounded-lg px-3 py-2 max-w-xs text-sm ${
                      message.type === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-white/10 text-white border border-white/20'
                    }`}>
                      {message.content.substring(0, 200)}...
                    </div>
                  </div>
                ))}
              </div>
              <div className="border-t border-white/10 p-4">
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask about the code..."
                    className="flex-1 bg-white/5 border border-white/20 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    onClick={handleSubmit}
                    disabled={!inputValue.trim() || isTyping}
                    className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white p-2 rounded-lg transition-colors"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
