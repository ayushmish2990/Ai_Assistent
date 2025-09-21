import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = {
  analyzeCode: async (code, language = null, filePath = null) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/analyze`, { 
        code, 
        language, 
        file_path: filePath 
      });
      return response.data;
    } catch (error) {
      console.error('Error analyzing code:', error);
      throw error;
    }
  },
  
  sendChatMessage: async (message, conversationHistory = []) => {
    try {
      // Convert frontend message format to backend format
      const formattedHistory = (conversationHistory || []).map(msg => ({
        role: msg.sender === 'bot' ? 'assistant' : 'user',
        content: msg.text
      }));
      
      const response = await axios.post(`${API_BASE_URL}/api/v1/chat`, { 
        message, 
        conversation_history: formattedHistory 
      });
      return response.data;
    } catch (error) {
      console.error('Error sending chat message:', error);
      throw error;
    }
  },

  completeCode: async (code, cursorPosition, language = null, filePath = null) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/complete`, {
        code,
        cursor_position: cursorPosition,
        language,
        file_path: filePath
      });
      return response.data;
    } catch (error) {
      console.error('Error completing code:', error);
      throw error;
    }
  },

  generateCode: async (prompt, language, context = null, filePath = null) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/generate`, {
        prompt,
        language,
        context,
        file_path: filePath
      });
      return response.data;
    } catch (error) {
      console.error('Error generating code:', error);
      throw error;
    }
  },

  indexCodebase: async (projectPath, includePatterns = null, excludePatterns = null) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/codebase/index`, {
        project_path: projectPath,
        include_patterns: includePatterns,
        exclude_patterns: excludePatterns
      });
      return response.data;
    } catch (error) {
      console.error('Error indexing codebase:', error);
      throw error;
    }
  },

  searchFiles: async (query, fileTypes = null, maxResults = 10) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/codebase/search/files`, {
        query,
        file_types: fileTypes,
        max_results: maxResults
      });
      return response.data;
    } catch (error) {
      console.error('Error searching files:', error);
      throw error;
    }
  },

  searchCode: async (query, language = null, contextLines = 3, maxResults = 20) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/codebase/search/code`, {
        query,
        language,
        context_lines: contextLines,
        max_results: maxResults
      });
      return response.data;
    } catch (error) {
      console.error('Error searching code:', error);
      throw error;
    }
  },

  getCodebaseStatus: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/codebase/status`);
      return response.data;
    } catch (error) {
      console.error('Error getting codebase status:', error);
      throw error;
    }
  },

  // Debug and Error Detection
  analyzeCodeErrors: async (code, language = 'python', filePath = null) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/debug/analyze`, {
        code,
        language,
        file_path: filePath
      });
      return response.data;
    } catch (error) {
      console.error('Error analyzing code errors:', error);
      throw error;
    }
  },

  autoFixCode: async (code, language = 'python', filePath = null) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/debug/auto-fix`, {
        code,
        language,
        file_path: filePath
      });
      return response.data;
    } catch (error) {
      console.error('Error auto-fixing code:', error);
      throw error;
    }
  },

  getFixSuggestion: async (code, line, column, errorType, language = 'python') => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/debug/fix-suggestion`, {
        code,
        line,
        column,
        error_type: errorType,
        language
      });
      return response.data;
    } catch (error) {
      console.error('Error getting fix suggestion:', error);
      throw error;
    }
  },

  getSupportedLanguages: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/debug/supported-languages`);
      return response.data;
    } catch (error) {
      console.error('Error getting supported languages:', error);
      throw error;
    }
  },

  // Real-time Collaboration
  createCollaborationSession: async (name, userName, userEmail) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/collaboration/sessions`, {
        name,
        user_name: userName,
        user_email: userEmail
      });
      return response.data;
    } catch (error) {
      console.error('Error creating collaboration session:', error);
      throw error;
    }
  },

  joinCollaborationSession: async (sessionId, userName, userEmail) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/collaboration/sessions/join`, {
        session_id: sessionId,
        user_name: userName,
        user_email: userEmail
      });
      return response.data;
    } catch (error) {
      console.error('Error joining collaboration session:', error);
      throw error;
    }
  },

  getCollaborationSession: async (sessionId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/collaboration/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Error getting collaboration session:', error);
      throw error;
    }
  },

  getUserSessions: async (userId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/collaboration/users/${userId}/sessions`);
      return response.data;
    } catch (error) {
      console.error('Error getting user sessions:', error);
      throw error;
    }
  },

  // WebSocket connection for real-time collaboration
  createWebSocketConnection: (sessionId, userId) => {
    const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/api/v1/collaboration/ws/${sessionId}/${userId}`;
    return new WebSocket(wsUrl);
  }
};

export default api;