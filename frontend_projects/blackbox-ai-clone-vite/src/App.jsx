import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { useState, useEffect, Suspense, lazy } from 'react';
import { ThemeProvider } from '@/components/theme-provider';
import { Toaster } from '@/components/ui/toaster';
import { AuthProvider, useAuth } from '@/contexts/auth-context';
import ProtectedRoute from '@/components/protected-route';
import { Loader2, Menu, User, Settings, LogOut } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { ModeToggle } from '@/components/mode-toggle';

// Lazy load pages
const Login = lazy(() => import('./pages/auth/login'));
const Register = lazy(() => import('./pages/auth/register'));
const ChatInterface = lazy(() => import('./components/chat-interface'));
const Sidebar = lazy(() => import('./components/sidebar'));
const UnauthorizedPage = lazy(() => import('./pages/unauthorized'));
const NotFoundPage = lazy(() => import('./pages/not-found'));

// Layout component
function DefaultLayout({ children }) {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      setSidebarOpen(!mobile);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);

  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="h-14 border-b flex items-center justify-between px-4 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="flex items-center">
            <button 
              onClick={toggleSidebar}
              className="mr-2 p-1 rounded-md hover:bg-accent md:hidden"
            >
              <Menu className="h-5 w-5" />
            </button>
            <h1 className="text-lg font-semibold">Blackbox AI</h1>
          </div>
          
          <div className="flex items-center space-x-2">
            <ModeToggle />
            {user && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                    <Avatar className="h-8 w-8">
                      <AvatarImage src={user.avatar} alt={user.name || user.email} />
                      <AvatarFallback>
                        {user.name
                          ? user.name.split(' ').map(n => n[0]).join('')
                          : user.email[0].toUpperCase()}
                      </AvatarFallback>
                    </Avatar>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56" align="end" forceMount>
                  <DropdownMenuLabel className="font-normal">
                    <div className="flex flex-col space-y-1">
                      <p className="text-sm font-medium leading-none">{user.name || 'User'}</p>
                      <p className="text-xs leading-none text-muted-foreground">
                        {user.email}
                      </p>
                    </div>
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => navigate('/profile')}>
                    <User className="mr-2 h-4 w-4" />
                    <span>Profile</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => navigate('/settings')}>
                    <Settings className="mr-2 h-4 w-4" />
                    <span>Settings</span>
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={logout}>
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Log out</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>
        </header>
        
        <div className="flex-1 overflow-auto">
          <Suspense fallback={
            <div className="flex items-center justify-center h-full">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          }>
            {children}
          </Suspense>
        </div>
      </main>
    </div>
  );
}

// Loading component
const LoadingFallback = () => (
  <div className="flex items-center justify-center min-h-screen">
    <Loader2 className="h-8 w-8 animate-spin text-primary" />
  </div>
);

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" storageKey="vite-ui-theme">
      <AuthProvider>
        <Router>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              {/* Public routes */}
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              
              {/* Protected routes */}
              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <DefaultLayout>
                      <ChatInterface />
                    </DefaultLayout>
                  </ProtectedRoute>
                }
              />
              
              {/* Admin routes */}
              <Route
                path="/admin/*"
                element={
                  <ProtectedRoute requireAdmin>
                    <DefaultLayout>
                      <AdminRoutes />
                    </DefaultLayout>
                  </ProtectedRoute>
                }
              />
              
              {/* Error pages */}
              <Route path="/unauthorized" element={<UnauthorizedPage />} />
              <Route path="/404" element={<NotFoundPage />} />
              <Route path="*" element={<Navigate to="/404" replace />} />
            </Routes>
          </Suspense>
          <Toaster />
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
