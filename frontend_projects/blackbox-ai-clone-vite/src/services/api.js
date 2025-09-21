import axios from 'axios';
import { ENDPOINTS, STORAGE_KEYS } from '../config';

// Create axios instance with base URL
const api = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:3001/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor to add the auth token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem(STORAGE_KEYS.AUTH_TOKEN);
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor to handle token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // If the error status is 401 and we haven't already tried to refresh the token
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem(STORAGE_KEYS.REFRESH_TOKEN);
        if (!refreshToken) {
          // No refresh token available, log the user out
          clearAuthData();
          window.location.href = '/login';
          return Promise.reject(error);
        }

        // Try to refresh the token
        const response = await axios.post(`${ENDPOINTS.AUTH.REFRESH}`, {
          refreshToken,
        });

        const { token, refreshToken: newRefreshToken } = response.data;
        
        // Update tokens in storage
        localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
        localStorage.setItem(STORAGE_KEYS.REFRESH_TOKEN, newRefreshToken);

        // Update the authorization header
        originalRequest.headers.Authorization = `Bearer ${token}`;

        // Retry the original request
        return api(originalRequest);
      } catch (error) {
        // If refresh token is invalid, clear auth data and redirect to login
        clearAuthData();
        window.location.href = '/login';
        return Promise.reject(error);
      }
    }

    return Promise.reject(error);
  }
);

// Helper function to clear authentication data
const clearAuthData = () => {
  localStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
  localStorage.removeItem(STORAGE_KEYS.REFRESH_TOKEN);
  localStorage.removeItem(STORAGE_KEYS.USER);
};

// Auth API
export const authAPI = {
  login: async (email, password) => {
    const response = await api.post(ENDPOINTS.AUTH.LOGIN, { email, password });
    return response.data;
  },
  
  register: async (userData) => {
    const response = await api.post(ENDPOINTS.AUTH.REGISTER, userData);
    return response.data;
  },
  
  getProfile: async () => {
    const response = await api.get(ENDPOINTS.AUTH.PROFILE);
    return response.data;
  },
  
  logout: async () => {
    try {
      await api.post(ENDPOINTS.AUTH.LOGOUT);
    } finally {
      clearAuthData();
    }
  },
};

// Chat API
export const chatAPI = {
  getConversations: async () => {
    const response = await api.get(ENDPOINTS.CHAT.CONVERSATIONS);
    return response.data;
  },
  
  getMessages: async (conversationId, page = 1, limit = 20) => {
    const response = await api.get(ENDPOINTS.CHAT.MESSAGES(conversationId), {
      params: { page, limit },
    });
    return response.data;
  },
  
  sendMessage: async (conversationId, content) => {
    const response = await api.post(ENDPOINTS.CHAT.SEND_MESSAGE(conversationId), {
      content,
    });
    return response.data;
  },
  
  createConversation: async (title = 'New Chat') => {
    const response = await api.post(ENDPOINTS.CHAT.CONVERSATIONS, { title });
    return response.data;
  },
};

export default api;
