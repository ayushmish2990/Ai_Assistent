import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { authAPI } from '@/services/api';
import { STORAGE_KEYS } from '@/config';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();

  // Load user from localStorage on mount
  useEffect(() => {
    const loadUser = async () => {
      try {
        const storedUser = localStorage.getItem(STORAGE_KEYS.USER);
        const token = localStorage.getItem(STORAGE_KEYS.AUTH_TOKEN);
        
        if (token && storedUser) {
          // If we have a token but no user in state, fetch the user
          setUser(JSON.parse(storedUser));
          
          // Verify token is still valid
          try {
            await authAPI.getProfile();
          } catch (err) {
            // Token is invalid, clear auth data
            await logout();
          }
        }
      } catch (err) {
        console.error('Failed to load user:', err);
        await logout();
      } finally {
        setLoading(false);
      }
    };

    loadUser();
  }, []);

  // Save user to localStorage when it changes
  useEffect(() => {
    if (user) {
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(user));
    }
  }, [user]);

  const login = async (email, password) => {
    try {
      setError(null);
      const { user: userData, token, refreshToken } = await authAPI.login(email, password);
      
      // Store tokens and user data
      localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
      localStorage.setItem(STORAGE_KEYS.REFRESH_TOKEN, refreshToken);
      setUser(userData);
      
      // Redirect to the original page or home
      const from = location.state?.from?.pathname || '/';
      navigate(from, { replace: true });
      
      return userData;
    } catch (err) {
      console.error('Login error:', err);
      setError(err.response?.data?.message || 'Login failed. Please try again.');
      throw err;
    }
  };

  const register = async (userData) => {
    try {
      setError(null);
      const { user: newUser, token, refreshToken } = await authAPI.register(userData);
      
      // Store tokens and user data
      localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
      localStorage.setItem(STORAGE_KEYS.REFRESH_TOKEN, refreshToken);
      setUser(newUser);
      
      // Redirect to home after registration
      navigate('/');
      
      return newUser;
    } catch (err) {
      console.error('Registration error:', err);
      setError(err.response?.data?.message || 'Registration failed. Please try again.');
      throw err;
    }
  };

  const logout = useCallback(async () => {
    try {
      await authAPI.logout();
    } catch (err) {
      console.error('Logout error:', err);
    } finally {
      // Clear all auth data
      Object.values(STORAGE_KEYS).forEach(key => {
        localStorage.removeItem(key);
      });
      setUser(null);
      navigate('/login');
    }
  }, [navigate]);

  const updateUser = useCallback((updates) => {
    setUser(prev => ({
      ...prev,
      ...updates,
    }));
  }, []);

  const value = {
    user,
    loading,
    error,
    setError,
    login,
    register,
    logout,
    updateUser,
    isAuthenticated: !!user,
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
