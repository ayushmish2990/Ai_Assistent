import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/auth-context';
import { useForm } from '@/hooks/use-form';
import { registerSchema } from '@/lib/validations';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Icons } from '@/components/icons';
import { toast } from '@/components/ui/use-toast';

export function RegisterPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const { register: registerUser, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  // Redirect if already authenticated
  React.useEffect(() => {
    if (isAuthenticated) {
      navigate('/', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  const { register, handleSubmit, formState: { errors }, watch } = useForm({
    schema: registerSchema,
    defaultValues: {
      username: '',
      email: '',
      password: '',
      confirmPassword: '',
      terms: false,
    },
  });

  const password = watch('password');

  const onSubmit = async (data) => {
    try {
      setIsLoading(true);
      await registerUser({
        username: data.username,
        email: data.email,
        password: data.password,
      });
      
      toast({
        title: 'Welcome!',
        description: 'Your account has been created successfully.',
      });
      
      // Redirect handled by the effect above
    } catch (error) {
      console.error('Registration error:', error);
      // Error handling is done in the useForm hook
    } finally {
      setIsLoading(false);
    }
  };

  const renderPasswordStrength = () => {
    if (!password) return null;
    
    let strength = 0;
    let messages = [];
    
    if (password.length >= 8) strength++;
    if (/[A-Z]/.test(password)) strength++;
    if (/[0-9]/.test(password)) strength++;
    if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) strength++;
    
    return (
      <div className="mt-1">
        <div className="flex space-x-1 mb-1">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className={`h-1 flex-1 rounded-full ${
                i <= strength ? 'bg-green-500' : 'bg-gray-200 dark:bg-gray-700'
              }`}
            />
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          {strength < 2 ? 'Weak' : strength < 4 ? 'Good' : 'Strong'} password
        </p>
      </div>
    );
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
        <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[400px]">
          <div className="flex flex-col space-y-2 text-center">
            <h1 className="text-2xl font-semibold tracking-tight">
              Create an account
            </h1>
            <p className="text-sm text-muted-foreground">
              Enter your details to get started
            </p>
          </div>
          
          <div className="grid gap-6">
            <form onSubmit={handleSubmit(onSubmit)}>
              <div className="grid gap-4">
                <div className="grid gap-1">
                  <Label htmlFor="username">Username</Label>
                  <Input
                    id="username"
                    placeholder="johndoe"
                    autoCapitalize="none"
                    autoComplete="username"
                    autoCorrect="off"
                    disabled={isLoading}
                    {...register('username')}
                  />
                  {errors.username && (
                    <p className="px-1 text-xs text-red-600">
                      {errors.username.message}
                    </p>
                  )}
                </div>
                
                <div className="grid gap-1">
                  <Label htmlFor="email">Email</Label>
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
                  <Label htmlFor="password">Password</Label>
                  <div className="relative">
                    <Input
                      id="password"
                      placeholder="••••••••"
                      type={showPassword ? 'text' : 'password'}
                      autoComplete="new-password"
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
                  {renderPasswordStrength()}
                  {errors.password && (
                    <p className="px-1 text-xs text-red-600">
                      {errors.password.message}
                    </p>
                  )}
                </div>
                
                <div className="grid gap-1">
                  <Label htmlFor="confirmPassword">Confirm Password</Label>
                  <div className="relative">
                    <Input
                      id="confirmPassword"
                      placeholder="••••••••"
                      type={showConfirmPassword ? 'text' : 'password'}
                      autoComplete="new-password"
                      disabled={isLoading}
                      {...register('confirmPassword')}
                      className="pr-10"
                    />
                    <button
                      type="button"
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      tabIndex={-1}
                    >
                      {showConfirmPassword ? (
                        <Icons.eyeOff className="h-4 w-4" />
                      ) : (
                        <Icons.eye className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                  {errors.confirmPassword && (
                    <p className="px-1 text-xs text-red-600">
                      {errors.confirmPassword.message}
                    </p>
                  )}
                </div>
                
                <div className="flex items-start space-x-2 mt-2">
                  <Checkbox
                    id="terms"
                    disabled={isLoading}
                    {...register('terms')}
                  />
                  <div className="grid gap-1.5 leading-none">
                    <label
                      htmlFor="terms"
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      I agree to the{' '}
                      <Link to="/terms" className="underline underline-offset-4 hover:text-primary">
                        Terms of Service
                      </Link>{' '}
                      and{' '}
                      <Link to="/privacy" className="underline underline-offset-4 hover:text-primary">
                        Privacy Policy
                      </Link>
                    </label>
                    {errors.terms && (
                      <p className="text-xs text-red-600">
                        {errors.terms.message}
                      </p>
                    )}
                  </div>
                </div>
                
                <Button type="submit" disabled={isLoading} className="w-full mt-2">
                  {isLoading && (
                    <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                  )}
                  Create Account
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
            Already have an account?{' '}
            <Link
              to="/login"
              className="underline underline-offset-4 hover:text-primary"
            >
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default RegisterPage;
