// API Configuration
export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:3001/api';

export const ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    REGISTER: '/auth/register',
    PROFILE: '/auth/me',
    REFRESH: '/auth/refresh',
    LOGOUT: '/auth/logout',
  },
  CHAT: {
    CONVERSATIONS: '/chat/conversations',
    MESSAGES: (conversationId) => `/chat/conversations/${conversationId}/messages`,
    SEND_MESSAGE: (conversationId) => `/chat/conversations/${conversationId}/messages`,
  },
};

// Local Storage Keys
export const STORAGE_KEYS = {
  AUTH_TOKEN: 'auth_token',
  REFRESH_TOKEN: 'refresh_token',
  USER: 'user',
};

// Default settings
export const DEFAULT_SETTINGS = {
  THEME: 'system', // 'light', 'dark', or 'system'
  MESSAGES_PER_PAGE: 20,
  CODE_THEME: 'github-dark', // For syntax highlighting
  ENABLE_NOTIFICATIONS: true,
};

// Validation constants
export const VALIDATION = {
  PASSWORD: {
    MIN_LENGTH: 8,
    REQUIRE_NUMBER: true,
    REQUIRE_UPPERCASE: true,
    REQUIRE_SYMBOL: true,
  },
  USERNAME: {
    MIN_LENGTH: 3,
    MAX_LENGTH: 30,
  },
};

// UI Constants
export const UI = {
  SIDEBAR_WIDTH: '16rem',
  HEADER_HEIGHT: '3.5rem',
  TRANSITION_DURATION: '200ms',
  TOAST_DURATION: 5000, // ms
};

// Feature Flags
export const FEATURES = {
  ENABLE_REGISTRATION: true,
  ENABLE_PASSWORD_RESET: true,
  ENABLE_EMAIL_VERIFICATION: false,
  ENABLE_THEME_SWITCHER: true,
};
