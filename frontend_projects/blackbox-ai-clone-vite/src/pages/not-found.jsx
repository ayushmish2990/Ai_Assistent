import { FileQuestion } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useNavigate } from 'react-router-dom';

export default function NotFoundPage() {
  const navigate = useNavigate();

  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <div className="w-full max-w-md space-y-6 text-center">
        <div className="flex flex-col items-center space-y-4">
          <div className="rounded-full bg-primary/10 p-4">
            <FileQuestion className="h-12 w-12 text-primary" />
          </div>
          <h1 className="text-4xl font-bold tracking-tight">404</h1>
          <h2 className="text-2xl font-semibold">Page Not Found</h2>
          <p className="text-muted-foreground">
            The page you're looking for doesn't exist or has been moved.
          </p>
        </div>
        
        <div className="flex flex-col space-y-2 pt-4">
          <Button 
            onClick={() => navigate(-1)} 
            variant="outline"
            className="w-full"
          >
            Go Back
          </Button>
          <Button 
            onClick={() => navigate('/')} 
            className="w-full"
          >
            Return Home
          </Button>
        </div>
      </div>
    </div>
  );
}
