import { AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { useNavigate } from 'react-router-dom';

export default function UnauthorizedPage() {
  const navigate = useNavigate();

  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <div className="w-full max-w-md space-y-6">
        <div className="flex flex-col items-center space-y-2 text-center">
          <AlertCircle className="h-12 w-12 text-destructive" />
          <h1 className="text-3xl font-bold tracking-tight">Access Denied</h1>
          <p className="text-muted-foreground">
            You don't have permission to access this page.
          </p>
        </div>
        
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Unauthorized Access</AlertTitle>
          <AlertDescription>
            Your account doesn't have the required permissions to view this page.
            Please contact an administrator if you believe this is a mistake.
          </AlertDescription>
        </Alert>
        
        <div className="flex flex-col space-y-2">
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
