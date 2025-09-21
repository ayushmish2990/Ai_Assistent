import { Button } from '../components/ui/button';
import { useNavigate } from 'react-router-dom';

export default function NotFound() {
  const navigate = useNavigate();
  
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center px-4">
      <h1 className="text-6xl font-bold text-primary mb-4">404</h1>
      <h2 className="text-3xl font-semibold mb-2">Page Not Found</h2>
      <p className="text-muted-foreground mb-8 max-w-md">
        The page you're looking for doesn't exist or has been moved to a different location.
      </p>
      <div className="flex gap-4">
        <Button variant="outline" onClick={() => window.history.back()}>
          Go Back
        </Button>
        <Button onClick={() => navigate('/')}>
          Return Home
        </Button>
      </div>
    </div>
  );
}
