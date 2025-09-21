import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Loader2, Code, RefreshCw, CheckCircle, AlertCircle } from 'lucide-react';
import { useToast } from '../ui/use-toast';

export function CodeReview() {
  const { t } = useTranslation();
  const { toast } = useToast();
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('javascript');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('issues');
  const [reviewResults, setReviewResults] = useState({
    issues: [],
    suggestions: [],
    refactoredCode: ''
  });

  const handleCodeReview = async () => {
    if (!code.trim()) {
      toast({
        title: t('errors.requiredField'),
        description: t('codeReview.errors.enterCode'),
        variant: 'destructive'
      });
      return;
    }

    setIsLoading(true);
    setActiveTab('issues');

    try {
      // TODO: Replace with actual API call to backend
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Mock response
      const mockResponse = {
        issues: [
          {
            id: 1,
            type: 'warning',
            message: 'Consider using const instead of let for variables that are not reassigned',
            line: 5,
            column: 7,
            codeSnippet: 'let counter = 0;'
          },
          {
            id: 2,
            type: 'error',
            message: 'Unused variable: unusedVar',
            line: 8,
            column: 5,
            codeSnippet: 'const unusedVar = 42;'
          }
        ],
        suggestions: [
          {
            id: 1,
            message: 'Consider using Array.map() for transforming arrays',
            line: 12,
            codeSnippet: 'const doubled = [];\nfor (let i = 0; i < numbers.length; i++) {\n  doubled.push(numbers[i] * 2);\n}'
          }
        ],
        refactoredCode: ''
      };

      setReviewResults(mockResponse);
    } catch (error) {
      console.error('Error performing code review:', error);
      toast({
        title: t('errors.error'),
        description: t('codeReview.errors.reviewFailed'),
        variant: 'destructive'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleRefactor = async () => {
    if (!code.trim()) {
      toast({
        title: t('errors.requiredField'),
        description: t('codeReview.errors.enterCode'),
        variant: 'destructive'
      });
      return;
    }

    setIsLoading(true);
    setActiveTab('refactored');

    try {
      // TODO: Replace with actual API call to backend
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock response
      const mockRefactored = {
        ...reviewResults,
        refactoredCode: '// Refactored code\nconst doubled = numbers.map(num => num * 2);\n\n// Fixed unused variable\nconst counter = 0;\n'
      };

      setReviewResults(mockRefactored);
    } catch (error) {
      console.error('Error refactoring code:', error);
      toast({
        title: t('errors.error'),
        description: t('codeReview.errors.refactorFailed'),
        variant: 'destructive'
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>{t('codeReview.title')}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="code-language" className="block text-sm font-medium mb-1">
                  {t('codeReview.labels.language')}
                </label>
                <select
                  id="code-language"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="w-full px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                  disabled={isLoading}
                >
                  <option value="javascript">JavaScript</option>
                  <option value="typescript">TypeScript</option>
                  <option value="python">Python</option>
                  <option value="java">Java</option>
                </select>
              </div>
              <div className="flex items-end gap-2">
                <Button 
                  onClick={handleCodeReview}
                  disabled={isLoading || !code.trim()}
                  className="flex-1 gap-2"
                >
                  {isLoading && activeTab !== 'refactored' ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Code className="h-4 w-4" />
                  )}
                  {t('codeReview.actions.reviewCode')}
                </Button>
                <Button 
                  onClick={handleRefactor}
                  disabled={isLoading || !code.trim()}
                  variant="outline"
                  className="gap-2"
                >
                  {isLoading && activeTab === 'refactored' ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                  {t('codeReview.actions.refactor')}
                </Button>
              </div>
            </div>
            
            <div>
              <label htmlFor="code-input" className="block text-sm font-medium mb-1">
                {t('codeReview.labels.codeInput')}
              </label>
              <Textarea
                id="code-input"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder={t('codeReview.placeholders.codeInput')}
                className="min-h-[200px] font-mono text-sm"
                disabled={isLoading}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {(reviewResults.issues.length > 0 || reviewResults.suggestions.length > 0 || reviewResults.refactoredCode) && (
        <Card className="border-0 shadow-sm">
          <CardHeader>
            <CardTitle>{t('codeReview.results.title')}</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="issues" disabled={reviewResults.issues.length === 0}>
                  <AlertCircle className="mr-2 h-4 w-4" />
                  {t('codeReview.tabs.issues')} {reviewResults.issues.length > 0 && `(${reviewResults.issues.length})`}
                </TabsTrigger>
                <TabsTrigger value="suggestions" disabled={reviewResults.suggestions.length === 0}>
                  <CheckCircle className="mr-2 h-4 w-4" />
                  {t('codeReview.tabs.suggestions')} {reviewResults.suggestions.length > 0 && `(${reviewResults.suggestions.length})`}
                </TabsTrigger>
                <TabsTrigger value="refactored" disabled={!reviewResults.refactoredCode}>
                  <RefreshCw className="mr-2 h-4 w-4" />
                  {t('codeReview.tabs.refactored')}
                </TabsTrigger>
              </TabsList>
              
              <div className="mt-6">
                <TabsContent value="issues" className="space-y-4">
                  {reviewResults.issues.length > 0 ? (
                    <div className="space-y-4">
                      {reviewResults.issues.map((issue) => (
                        <div 
                          key={issue.id} 
                          className={`p-4 rounded-md border ${
                            issue.type === 'error' 
                              ? 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800' 
                              : 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800'
                          }`}
                        >
                          <div className="flex items-start">
                            <div className="flex-shrink-0">
                              {issue.type === 'error' ? (
                                <AlertCircle className="h-5 w-5 text-red-500 dark:text-red-400" />
                              ) : (
                                <AlertCircle className="h-5 w-5 text-yellow-500 dark:text-yellow-400" />
                              )}
                            </div>
                            <div className="ml-3">
                              <div className="text-sm font-medium">
                                {issue.type === 'error' ? 'Error' : 'Warning'} on line {issue.line}, column {issue.column}
                              </div>
                              <div className="mt-1 text-sm">
                                {issue.message}
                              </div>
                              {issue.codeSnippet && (
                                <div className="mt-2 p-2 bg-gray-100 dark:bg-gray-800 rounded text-xs font-mono">
                                  {issue.codeSnippet}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      {t('codeReview.messages.noIssues')}
                    </div>
                  )}
                </TabsContent>
                
                <TabsContent value="suggestions" className="space-y-4">
                  {reviewResults.suggestions.length > 0 ? (
                    <div className="space-y-4">
                      {reviewResults.suggestions.map((suggestion) => (
                        <div 
                          key={suggestion.id} 
                          className="p-4 rounded-md border bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800"
                        >
                          <div className="flex items-start">
                            <div className="flex-shrink-0">
                              <CheckCircle className="h-5 w-5 text-blue-500 dark:text-blue-400" />
                            </div>
                            <div className="ml-3">
                              <div className="text-sm font-medium">
                                Suggestion for line {suggestion.line}
                              </div>
                              <div className="mt-1 text-sm">
                                {suggestion.message}
                              </div>
                              {suggestion.codeSnippet && (
                                <div className="mt-2 p-2 bg-gray-100 dark:bg-gray-800 rounded text-xs font-mono whitespace-pre">
                                  {suggestion.codeSnippet}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      {t('codeReview.messages.noSuggestions')}
                    </div>
                  )}
                </TabsContent>
                
                <TabsContent value="refactored" className="space-y-4">
                  {reviewResults.refactoredCode ? (
                    <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-md border">
                      <pre className="text-sm font-mono whitespace-pre-wrap">{reviewResults.refactoredCode}</pre>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      {t('codeReview.messages.noRefactored')}
                    </div>
                  )}
                </TabsContent>
              </div>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
