import { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/auth-context';
import { useForm } from '@/hooks/use-form';
import { loginSchema } from '@/lib/validations';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Icons } from '@/components/icons';
import { toast } from '@/components/ui/use-toast';

export function LoginPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Redirect if already authenticated
  React.useEffect(() => {
    if (isAuthenticated) {
      navigate(location.state?.from?.pathname || '/', { replace: true });
    }
  }, [isAuthenticated, navigate, location]);

  const { register, handleSubmit, formState: { errors } } = useForm({
    schema: loginSchema,
    defaultValues: {
      email: '',
      password: '',
      rememberMe: false,
    },
  });

  const onSubmit = async (data) => {
    try {
      setIsLoading(true);
      await login(data.email, data.password);
      
      toast({
        title: 'Welcome back!',
        description: 'You have successfully logged in.',
      });
      
      // Redirect handled by the effect above
    } catch (error) {
      console.error('Login error:', error);
      // Error handling is done in the useForm hook
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container relative h-screen flex-col items-center justify-center md:grid lg:max-w-none lg:grid-cols-2 lg:px-0">
      <div className="relative hidden h-full flex-col bg-muted p-10 text-white lg:flex dark:border-r">
        <div className="absolute inset-0 bg-zinc-900" />
        <div className="relative z-20 flex items-center text-lg font-medium">
          <Icons.logo className="mr-2 h-6 w-6" />
          Blackbox AI
        </div>
        <div className="relative z-20 mt-auto">
          <blockquote className="space-y-2">
            <p className="text-lg">
              "The most powerful AI coding assistant at your fingertips."
            </p>
            <footer className="text-sm">Join thousands of developers</footer>
          </blockquote>
        </div>
      </div>
      
      <div className="lg:p-8">
        <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[350px]">
          <div className="flex flex-col space-y-2 text-center">
            <h1 className="text-2xl font-semibold tracking-tight">
              Welcome back
            </h1>
            <p className="text-sm text-muted-foreground">
              Enter your email and password to sign in
            </p>
          </div>
          
          <div className="grid gap-6">
            <form onSubmit={handleSubmit(onSubmit)}>
              <div className="grid gap-4">
                <div className="grid gap-1">
                  <Label htmlFor="email" className="mb-1">
                    Email
                  </Label>
                  <Input
                    id="email"
                    placeholder="name@example.com"
                    type="email"
                    autoCapitalize="none"
                    autoComplete="email"
                    autoCorrect="off"
                    disabled={isLoading}
                    {...register('email')}
                  />
                  {errors.email && (
                    <p className="px-1 text-xs text-red-600">
                      {errors.email.message}
                    </p>
                  )}
                </div>
                
                <div className="grid gap-1">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="password">Password</Label>
                    <Link
                      to="/forgot-password"
                      className="text-sm font-medium text-primary hover:underline"
                    >
                      Forgot password?
                    </Link>
                  </div>
                  <div className="relative">
                    <Input
                      id="password"
                      placeholder="••••••••"
                      type={showPassword ? 'text' : 'password'}
                      autoComplete="current-password"
                      disabled={isLoading}
                      {...register('password')}
                      className="pr-10"
                    />
                    <button
                      type="button"
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                      onClick={() => setShowPassword(!showPassword)}
                      tabIndex={-1}
                    >
                      {showPassword ? (
                        <Icons.eyeOff className="h-4 w-4" />
                      ) : (
                        <Icons.eye className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                  {errors.password && (
                    <p className="px-1 text-xs text-red-600">
                      {errors.password.message}
                    </p>
                  )}
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="remember-me"
                      disabled={isLoading}
                      {...register('rememberMe')}
                    />
                    <Label htmlFor="remember-me" className="text-sm font-medium">
                      Remember me
                    </Label>
                  </div>
                </div>
                
                <Button type="submit" disabled={isLoading} className="w-full">
                  {isLoading && (
                    <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                  )}
                  Sign In
                </Button>
              </div>
            </form>
            
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">
                  Or continue with
                </span>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <Button variant="outline" type="button" disabled={isLoading}>
                {isLoading ? (
                  <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Icons.gitHub className="mr-2 h-4 w-4" />
                )}
                GitHub
              </Button>
              <Button variant="outline" type="button" disabled={isLoading}>
                {isLoading ? (
                  <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Icons.google className="mr-2 h-4 w-4" />
                )}
                Google
              </Button>
            </div>
          </div>
          
          <p className="px-8 text-center text-sm text-muted-foreground">
            Don't have an account?{' '}
            <Link
              to="/register"
              className="underline underline-offset-4 hover:text-primary"
            >
              Sign up
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default LoginPage;
