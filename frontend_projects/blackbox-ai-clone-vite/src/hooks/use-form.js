import { useForm as useHookForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { toast } from '@/components/ui/use-toast';

export function useForm({ schema, defaultValues = {}, onSubmit, onError } = {}) {
  const form = useHookForm({
    resolver: schema ? zodResolver(schema) : undefined,
    defaultValues,
    mode: 'onBlur',
  });

  const handleSubmit = form.handleSubmit(
    async (data) => {
      try {
        await onSubmit(data);
      } catch (error) {
        console.error('Form submission error:', error);
        
        // Handle form-level errors
        if (error.response?.data?.errors) {
          // Handle field-specific errors
          Object.entries(error.response.data.errors).forEach(([field, message]) => {
            form.setError(field, {
              type: 'server',
              message: Array.isArray(message) ? message[0] : message,
            });
          });
        } else {
          // Show generic error message
          toast({
            title: 'Error',
            description: error.response?.data?.message || 'An error occurred. Please try again.',
            variant: 'destructive',
          });
        }
        
        // Call the onError callback if provided
        if (onError) {
          onError(error);
        }
      }
    },
    (errors) => {
      // Handle validation errors
      console.error('Form validation errors:', errors);
      
      // Show a toast for the first error
      const firstError = Object.values(errors)[0];
      if (firstError?.message) {
        toast({
          title: 'Validation Error',
          description: firstError.message,
          variant: 'destructive',
        });
      }
    }
  );

  return {
    ...form,
    handleSubmit,
    register: form.register,
    errors: form.formState.errors,
    isSubmitting: form.formState.isSubmitting,
    isDirty: form.formState.isDirty,
    isValid: form.formState.isValid,
    reset: form.reset,
    setValue: form.setValue,
    getValues: form.getValues,
    watch: form.watch,
    control: form.control,
  };
}
