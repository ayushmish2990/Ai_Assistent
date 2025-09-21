import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Copy, Loader2, Wrench, CheckCircle2 } from 'lucide-react';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { toast } from '../ui/use-toast';
import { Badge } from '../ui/badge';

export function TestGenerator() {
  const { t } = useTranslation();
  const { toast } = useToast();
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('javascript');
  const [testFramework, setTestFramework] = useState('jest');
  const [isLoading, setIsLoading] = useState(false);
  const [isFixing, setIsFixing] = useState(false);
  const [testCode, setTestCode] = useState('');
  const [fixedCode, setFixedCode] = useState('');
  const [coverage, setCoverage] = useState(null);
  const [testResults, setTestResults] = useState(null);
  const [activeTab, setActiveTab] = useState('tests');
  const [showFixTab, setShowFixTab] = useState(false);

  const supportedLanguages = [
    { value: 'javascript', label: 'JavaScript' },
    { value: 'typescript', label: 'TypeScript' },
    { value: 'python', label: 'Python' },
    { value: 'java', label: 'Java' },
  ];

  const testFrameworks = {
    javascript: [
      { value: 'jest', label: 'Jest' },
      { value: 'mocha', label: 'Mocha' },
      { value: 'jasmine', label: 'Jasmine' },
    ],
    typescript: [
      { value: 'jest', label: 'Jest' },
      { value: 'mocha', label: 'Mocha' },
      { value: 'jasmine', label: 'Jasmine' },
    ],
    python: [
      { value: 'pytest', label: 'pytest' },
      { value: 'unittest', label: 'unittest' },
    ],
    java: [
      { value: 'junit', label: 'JUnit' },
      { value: 'testng', label: 'TestNG' },
    ],
  };

  const generateTests = async () => {
    if (!code.trim()) {
      toast({
        title: t('errors.requiredField'),
        description: t('tests.errors.enterCode'),
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(true);
    setActiveTab('tests');

    try {
      // TODO: Replace with actual API call to backend
      await new Promise((resolve) => setTimeout(resolve, 1500));

      // Mock response based on language and framework
      let mockTests = '';
      const functionName = code.match(/function\s+(\w+)/)?.[1] || 'example';
      const className = code.match(/class\s+(\w+)/)?.[1] || 'Example';

      switch (language) {
        case 'javascript':
        case 'typescript':
          mockTests = `// ${t('tests.mock.description', { functionName })}
describe('${className}', () => {
  test('${t('tests.mock.shouldWork', { functionName })}', () => {
    const result = ${functionName}('test', 42);
    expect(result).toBeDefined();
    // ${t('tests.mock.addMoreTests')}
  });
});`;
          break;
        case 'python':
          mockTests = `# ${t('tests.mock.description', { functionName })}
import unittest

class Test${className}(unittest.TestCase):
    def test_${functionName}(self):
        """${t('tests.mock.shouldWork', { functionName })}"""
        result = ${functionName}('test', 42)
        self.assertIsNotNone(result)
        # ${t('tests.mock.addMoreTests')}`;
          break;
        case 'java':
          mockTests = `// ${t('tests.mock.description', { functionName })}
import org.junit.Test;
import static org.junit.Assert.*;

public class ${className}Test {
    @Test
    public void test${functionName.charAt(0).toUpperCase() + functionName.slice(1)}() {
        // ${t('tests.mock.shouldWork', { functionName })}
        String result = ${className}.${functionName}("test", 42);
        assertNotNull(result);
        // ${t('tests.mock.addMoreTests')}
    }
}`;
          break;
        default:
          mockTests = t('tests.mock.unsupported');
      }

      setTestCode(mockTests);

      // Mock coverage data
      setCoverage({
        statements: Math.floor(Math.random() * 30) + 70, // 70-100%
        branches: Math.floor(Math.random() * 40) + 50,   // 50-90%
        functions: Math.floor(Math.random() * 35) + 60,  // 60-95%
        lines: Math.floor(Math.random() * 30) + 70,      // 70-100%
      });
    } catch (error) {
      console.error('Error generating tests:', error);
      toast({
        title: t('errors.error'),
        description: t('tests.errors.generationFailed'),
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunTests = async () => {
    setIsLoading(true);
    try {
      // Simulate test execution
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Mock test results (30% chance of failure for demo purposes)
      const hasErrors = Math.random() < 0.3;

      const mockCoverage = {
        statements: { covered: hasErrors ? 65 : 95, total: 100, percentage: hasErrors ? 65 : 95 },
        branches: { covered: hasErrors ? 50 : 90, total: 100, percentage: hasErrors ? 50 : 90 },
        functions: { covered: hasErrors ? 70 : 98, total: 100, percentage: hasErrors ? 70 : 98 },
        lines: { covered: hasErrors ? 67 : 96, total: 100, percentage: hasErrors ? 67 : 96 }
      };

      const mockTestResults = {
        passed: hasErrors ? 3 : 5,
        failed: hasErrors ? 2 : 0,
        total: 5,
        errors: hasErrors ? [
          { test: 'should handle edge cases', message: 'Expected 5 but got 3' },
          { test: 'should validate input', message: 'TypeError: Cannot read property of undefined' }
        ] : []
      };

      setCoverage(mockCoverage);
      setTestResults(mockTestResults);
      setShowFixTab(hasErrors);

      if (hasErrors) {
        toast.error(t('tests.testsFailed', { failed: mockTestResults.failed }));
      } else {
        toast.success(t('tests.testsPassed'));
      }
    } catch (error) {
      toast.error(t('tests.testError'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleFixErrors = async () => {
    if (!testResults?.errors?.length) return;

    setIsFixing(true);
    try {
      // Simulate AI fixing errors
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Generate mock fixed code based on the original code and errors
      const fixed = `// Fixed version of the code\n${code}\n\n// Fixed issues:\n${testResults.errors.map((err, i) => `// - ${err.test}: ${err.message}`).join('\n')}`;

      setFixedCode(fixed);
      toast.success(t('tests.fixSuccess'));
    } catch (error) {
      toast.error(t('tests.fixError'));
    } finally {
      setIsFixing(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(testCode);
    toast({
      title: t('tests.copied'),
      description: t('tests.copiedToClipboard'),
    });
  };

  const getCoverageColor = (percentage) => {
    if (percentage >= 90) return 'bg-green-500';
    if (percentage >= 70) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-6">
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>{t('tests.title')}</CardTitle>
          <CardDescription>{t('tests.subtitle')}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label htmlFor="test-language" className="block text-sm font-medium mb-1">
                  {t('tests.labels.language')}
                </label>
                <Select 
                  value={language} 
                  onValueChange={(value) => {
                    setLanguage(value);
                    // Reset test framework to first available for the language
                    setTestFramework(testFrameworks[value]?.[0]?.value || '');
                  }}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder={t('tests.placeholders.language')} />
                  </SelectTrigger>
                  <SelectContent>
                    {supportedLanguages.map((lang) => (
                      <SelectItem key={lang.value} value={lang.value}>
                        {lang.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label htmlFor="test-framework" className="block text-sm font-medium mb-1">
                  {t('tests.labels.framework')}
                </label>
                <Select 
                  value={testFramework} 
                  onValueChange={setTestFramework}
                  disabled={!testFrameworks[language]?.length}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder={t('tests.placeholders.framework')} />
                  </SelectTrigger>
                  <SelectContent>
                    {testFrameworks[language]?.map((framework) => (
                      <SelectItem key={framework.value} value={framework.value}>
                        {framework.label}
                      </SelectItem>
                    )) || (
                      <SelectItem value="" disabled>
                        {t('tests.noFrameworks')}
                      </SelectItem>
                    )}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-end">
                <Button 
                  onClick={generateTests}
                  disabled={isLoading || !code.trim() || !testFrameworks[language]?.length}
                  className="w-full gap-2"
                >
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <FileCode className="h-4 w-4" />
                  )}
                  {t('tests.actions.generate')}
                </Button>
              </div>
            </div>

            <div>
              <label htmlFor="code-input" className="block text-sm font-medium mb-1">
                {t('tests.labels.codeInput')}
              </label>
              <Textarea
                id="code-input"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder={t('tests.placeholders.codeInput')}
                className="min-h-[200px] font-mono text-sm"
                disabled={isLoading}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {testCode && (
        <Card className="border-0 shadow-sm">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle>{t('tests.results.title')}</CardTitle>
              <div className="flex gap-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleRunTests}
                  className="gap-2"
                >
                  <Play className="h-4 w-4" />
                  {t('tests.actions.runTests')}
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={copyToClipboard}
                  className="gap-2"
                >
                  <Copy className="h-4 w-4" />
                  {t('tests.actions.copy')}
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="tests">{t('tests.tabs.tests')}</TabsTrigger>
                <TabsTrigger value="coverage">{t('tests.tabs.coverage')}</TabsTrigger>
                {showFixTab && (
                  <TabsTrigger value="fix">
                    {t('tests.tabs.fix')} {testResults?.failed > 0 && `(${testResults.failed})`}
                  </TabsTrigger>
                )}
              </TabsList>

              <div className="mt-4">
                <TabsContent value="tests">
                  <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-md border overflow-auto">
                    <pre className="whitespace-pre-wrap font-mono text-sm">
                      {testCode}
                    </pre>
                  </div>
                </TabsContent>

                <TabsContent value="coverage" className="mt-4">
                  {coverage ? (
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <h3 className="text-lg font-medium">{t('tests.coverage.legend')}</h3>
                        {testResults && (
                          <div className="text-sm">
                            <span className={testResults.failed > 0 ? 'text-destructive' : 'text-green-600'}>
                              {testResults.passed} {t('tests.passed')}
                            </span>
                            {testResults.failed > 0 && (
                              <span className="ml-2 text-destructive">
                                {testResults.failed} {t('tests.failed')}
                              </span>
                            )}
                            <span className="mx-2">•</span>
                            <span>{testResults.total} {t('tests.total')}</span>
                          </div>
                        )}
                      </div>
                      <div className="grid gap-4 md:grid-cols-2">
                        <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-md border">
                          <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            {t('tests.coverage.statements')}
                          </div>
                          <div className="text-2xl font-bold mt-1">
                            {coverage.statements}%
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                            <div 
                              className={`h-2 rounded-full ${getCoverageColor(coverage.statements)}`}
                              style={{ width: `${coverage.statements}%` }}
                            />
                          </div>
                        </div>

                        <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-md border">
                          <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            {t('tests.coverage.branches')}
                          </div>
                          <div className="text-2xl font-bold mt-1">
                            {coverage.branches}%
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                            <div 
                              className={`h-2 rounded-full ${getCoverageColor(coverage.branches)}`}
                              style={{ width: `${coverage.branches}%` }}
                            />
                          </div>
                        </div>
                        
                        <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-md border">
                          <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            {t('tests.coverage.functions')}
                          </div>
                          <div className="text-2xl font-bold mt-1">
                            {coverage.functions}%
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                            <div 
                              className={`h-2 rounded-full ${getCoverageColor(coverage.functions)}`}
                              style={{ width: `${coverage.functions}%` }}
                            />
                          </div>
                        </div>
                        
                        <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-md border">
                          <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            {t('tests.coverage.lines')}
                          </div>
                          <div className="text-2xl font-bold mt-1">
                            {coverage.lines}%
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                            <div 
                              className={`h-2 rounded-full ${getCoverageColor(coverage.lines)}`}
                              style={{ width: `${coverage.lines}%` }}
                            />
                          </div>
                        </div>
                      </div>
                      
                      <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-md border">
                        <h3 className="font-medium mb-2">{t('tests.coverage.legend')}</h3>
                        <div className="flex flex-wrap gap-4">
                          <div className="flex items-center">
                            <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                            <span className="text-sm">≥ 90%</span>
                          </div>
                          <div className="flex items-center">
                            <div className="w-3 h-3 rounded-full bg-yellow-500 mr-2"></div>
                            <span className="text-sm">70-89%</span>
                          </div>
                          <div className="flex items-center">
                            <div className="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
                            <span className="text-sm">&lt; 70%</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      {t('tests.coverage.notAvailable')}
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
