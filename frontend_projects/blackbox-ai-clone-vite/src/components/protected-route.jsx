import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/auth-context';
import { Loader2 } from 'lucide-react';

/**
 * A component that renders children only if the user is authenticated.
 * If not authenticated, it will redirect to the login page.
 * 
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.children - The child components to render if authenticated
 * @param {boolean} [props.requireAdmin=false] - If true, requires admin privileges
 * @param {string} [props.redirectTo='/login'] - The path to redirect to if not authenticated
 * @returns {JSX.Element} The protected route component
 */
export default function ProtectedRoute({ 
  children, 
  requireAdmin = false, 
  redirectTo = '/login' 
}) {
  const { user, loading } = useAuth();
  const location = useLocation();

  // Show loading spinner while checking auth state
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  // If not authenticated, redirect to login with return url
  if (!user) {
    return (
      <Navigate 
        to={redirectTo} 
        state={{ from: location }} 
        replace 
      />
    );
  }

  // If admin required but user is not admin
  if (requireAdmin && !user.isAdmin) {
    return <Navigate to="/unauthorized" replace />;
  }

  // User is authenticated (and admin if required), render children
  return children;
}
