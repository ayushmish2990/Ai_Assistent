import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Loader2, Bug, Sparkles } from 'lucide-react';

export function DebugAssistant() {
  const { t } = useTranslation();
  const [code, setCode] = useState('');
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const analyzeCode = async () => {
    if (!code.trim()) {
      setError(t('debug.errors.emptyCode'));
      return;
    }

    setIsLoading(true);
    setError('');
    setResult('');

    try {
      // TODO: Replace with actual API call to your AI service
      // This is a mock implementation
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Mock response - in a real app, this would come from your AI service
      const mockResponse = {
        issues: [
          {
            type: 'warning',
            message: 'Potential memory leak in the loop',
            line: 5,
            suggestion: 'Consider using `useCallback` to memoize the function.'
          },
          {
            type: 'error',
            message: 'Unhandled promise rejection',
            line: 12,
            suggestion: 'Add a .catch() block to handle potential errors.'
          }
        ],
        summary: 'Found 2 issues that need attention.'
      };

      setResult(mockResponse);
    } catch (err) {
      setError(t('debug.errors.analysisFailed'));
      console.error('Debug analysis failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const fixCode = async () => {
    if (!code.trim()) {
      setError(t('debug.errors.emptyCode'));
      return;
    }

    setIsLoading(true);
    setError('');
    setResult('');

    try {
      // TODO: Replace with actual API call to your AI service
      // This is a mock implementation
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock response - in a real app, this would come from your AI service
      const mockResponse = {
        fixedCode: code.replace(
          '// TODO: Fix this function', 
          '// Fixed: Added proper error handling and memoization\nconst processData = useCallback(async (data) => {\n  try {\n    const result = await fetchData(data);\n    return result;\n  } catch (error) {\n    console.error(\'Error processing data:\', error);\n    throw error;\n  }\n}, [fetchData]);'
        ),
        explanation: 'Added proper error handling with try/catch and memoized the function with useCallback to prevent unnecessary re-renders.'
      };

      setResult(mockResponse);
    } catch (err) {
      setError(t('debug.errors.fixFailed'));
      console.error('Code fixing failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">{t('debug.title')}</h2>
        <div className="flex space-x-2">
          <Button 
            variant="outline" 
            onClick={analyzeCode}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t('common.analyzing')}
              </>
            ) : (
              <>
                <Bug className="mr-2 h-4 w-4" />
                {t('debug.actions.analyze')}
              </>
            )}
          </Button>
          <Button 
            onClick={fixCode}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t('common.fixing')}
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                {t('debug.actions.fixCode')}
              </>
            )}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="space-y-2">
          <label htmlFor="code-input" className="text-sm font-medium">
            {t('debug.labels.codeInput')}
          </label>
          <Textarea
            id="code-input"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            placeholder={t('debug.placeholders.codeInput')}
            className="min-h-[300px] font-mono text-sm"
            disabled={isLoading}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">
            {t('debug.labels.results')}
          </label>
          <Card className="h-full">
            <CardContent className="p-6 h-full">
              {isLoading ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 className="mr-2 h-6 w-6 animate-spin" />
                  <span>{t('common.loading')}</span>
                </div>
              ) : error ? (
                <div className="text-red-500">{error}</div>
              ) : result ? (
                <div className="space-y-4">
                  {result.issues ? (
                    <>
                      <h3 className="font-semibold">{result.summary}</h3>
                      <div className="space-y-4">
                        {result.issues.map((issue, index) => (
                          <div 
                            key={index}
                            className={`p-3 rounded-md ${
                              issue.type === 'error' 
                                ? 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
                                : 'bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800'
                            }`}
                          >
                            <div className="flex items-center">
                              <span className={`font-medium ${
                                issue.type === 'error' 
                                  ? 'text-red-700 dark:text-red-300'
                                  : 'text-yellow-700 dark:text-yellow-300'
                              }`}>
                                {issue.type === 'error' ? 'Error' : 'Warning'} on line {issue.line}
                              </span>
                            </div>
                            <p className="mt-1 text-sm">{issue.message}</p>
                            {issue.suggestion && (
                              <div className="mt-2 p-2 bg-white dark:bg-gray-800 rounded text-sm">
                                <p className="text-xs text-muted-foreground mb-1">Suggestion:</p>
                                <p>{issue.suggestion}</p>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </>
                  ) : result.fixedCode ? (
                    <>
                      <h3 className="font-semibold">{t('debug.results.fixedCode')}</h3>
                      <div className="mt-2 p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                        <pre className="text-sm whitespace-pre-wrap">{result.fixedCode}</pre>
                      </div>
                      {result.explanation && (
                        <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
                          <h4 className="font-medium text-blue-700 dark:text-blue-300">
                            {t('debug.results.explanation')}:
                          </h4>
                          <p className="mt-1 text-sm">{result.explanation}</p>
                        </div>
                      )}
                    </>
                  ) : null}
                </div>
              ) : (
                <div className="text-muted-foreground text-center h-full flex items-center justify-center">
                  {t('debug.messages.enterCode')}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
