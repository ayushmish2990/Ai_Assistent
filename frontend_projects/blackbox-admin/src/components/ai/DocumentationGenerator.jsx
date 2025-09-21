import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Loader2, FileText, Code, FileJson, FileType } from 'lucide-react';
import { useToast } from '../ui/use-toast';

export function DocumentationGenerator() {
  const { t } = useTranslation();
  const { toast } = useToast();
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('javascript');
  const [docType, setDocType] = useState('inline');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedDocs, setGeneratedDocs] = useState('');
  const [activeTab, setActiveTab] = useState('preview');

  const supportedLanguages = [
    { value: 'javascript', label: 'JavaScript' },
    { value: 'typescript', label: 'TypeScript' },
    { value: 'python', label: 'Python' },
    { value: 'java', label: 'Java' },
    { value: 'csharp', label: 'C#' },
  ];

  const docTypes = [
    { value: 'inline', label: 'Inline Comments', icon: <FileText className="h-4 w-4 mr-2" /> },
    { value: 'jsdoc', label: 'JSDoc', icon: <FileJson className="h-4 w-4 mr-2" /> },
    { value: 'readme', label: 'README', icon: <FileType className="h-4 w-4 mr-2" /> },
    { value: 'api', label: 'API Reference', icon: <Code className="h-4 w-4 mr-2" /> },
  ];

  const generateDocumentation = async () => {
    if (!code.trim()) {
      toast({
        title: t('errors.requiredField'),
        description: t('docs.errors.enterCode'),
        variant: 'destructive',
      });
      return;
    }

    setIsGenerating(true);
    setActiveTab('preview');

    try {
      // TODO: Replace with actual API call to backend
      await new Promise((resolve) => setTimeout(resolve, 1500));

      // Mock response based on doc type
      let mockResponse = '';
      const functionName = code.match(/function\s+(\w+)/)?.[1] || 'example';
      const className = code.match(/class\s+(\w+)/)?.[1] || 'Example';
      
      switch (docType) {
        case 'inline':
          mockResponse = `// ${t('docs.mock.inline', { className })}
${code}`;
          break;
        case 'jsdoc':
          mockResponse = `/**
 * ${t('docs.mock.jsdoc.title', { functionName })}
 * @param {string} param1 - ${t('docs.mock.jsdoc.param1')}
 * @param {number} param2 - ${t('docs.mock.jsdoc.param2')}
 * @returns {string} ${t('docs.mock.jsdoc.returns')}
 */
${code}`;
          break;
        case 'readme':
          mockResponse = `# ${className}

${t('docs.mock.readme.description', { className })}

## ${t('docs.mock.readme.installation')}

\`\`\`bash
npm install ${className.toLowerCase()}
\`\`\`

## ${t('docs.mock.readme.usage')}

\`\`\`javascript
// ${t('docs.mock.readme.example')}
import { ${functionName} } from '${className.toLowerCase()}';

const result = ${functionName}('test', 42);
console.log(result);
\`\`\`
`;
          break;
        case 'api':
          mockResponse = `# ${className} API Reference

## ${functionName}()

${t('docs.mock.api.description')}

### ${t('docs.mock.api.parameters')}

| Parameter | Type     | Description |
|-----------|----------|-------------|
| param1    | string   | ${t('docs.mock.api.param1')} |
| param2    | number   | ${t('docs.mock.api.param2')} |

### ${t('docs.mock.api.returns')}

${t('docs.mock.api.returnsText')}

### ${t('docs.mock.api.example')}

\`\`\`javascript
const result = ${functionName}('test', 42);
// ${t('docs.mock.api.exampleResult')}
\`\`\``;
          break;
        default:
          mockResponse = t('docs.mock.unsupported');
      }

      setGeneratedDocs(mockResponse);
    } catch (error) {
      console.error('Error generating documentation:', error);
      toast({
        title: t('errors.error'),
        description: t('docs.errors.generationFailed'),
        variant: 'destructive',
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(generatedDocs);
    toast({
      title: t('docs.copied'),
      description: t('docs.copiedToClipboard'),
    });
  };

  return (
    <div className="space-y-6">
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>{t('docs.title')}</CardTitle>
          <CardDescription>{t('docs.subtitle')}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label htmlFor="doc-language" className="block text-sm font-medium mb-1">
                  {t('docs.labels.language')}
                </label>
                <Select value={language} onValueChange={setLanguage}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder={t('docs.placeholders.language')} />
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
                <label htmlFor="doc-type" className="block text-sm font-medium mb-1">
                  {t('docs.labels.docType')}
                </label>
                <Select value={docType} onValueChange={setDocType}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder={t('docs.placeholders.docType')} />
                  </SelectTrigger>
                  <SelectContent>
                    {docTypes.map((type) => (
                      <SelectItem key={type.value} value={type.value}>
                        <div className="flex items-center">
                          {type.icon}
                          {type.label}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex items-end">
                <Button 
                  onClick={generateDocumentation}
                  disabled={isGenerating || !code.trim()}
                  className="w-full gap-2"
                >
                  {isGenerating ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <FileText className="h-4 w-4" />
                  )}
                  {t('docs.actions.generate')}
                </Button>
              </div>
            </div>
            
            <div>
              <label htmlFor="code-input" className="block text-sm font-medium mb-1">
                {t('docs.labels.codeInput')}
              </label>
              <Textarea
                id="code-input"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder={t('docs.placeholders.codeInput')}
                className="min-h-[200px] font-mono text-sm"
                disabled={isGenerating}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {generatedDocs && (
        <Card className="border-0 shadow-sm">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle>{t('docs.results.title')}</CardTitle>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={copyToClipboard}
                className="gap-2"
              >
                {t('docs.actions.copy')}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="preview">
                  {t('docs.tabs.preview')}
                </TabsTrigger>
                <TabsTrigger value="code">
                  {t('docs.tabs.raw')}
                </TabsTrigger>
              </TabsList>
              
              <div className="mt-4">
                <TabsContent value="preview">
                  <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-md border overflow-auto">
                    {docType === 'readme' || docType === 'api' ? (
                      <div 
                        className="prose dark:prose-invert max-w-none"
                        dangerouslySetInnerHTML={{ 
                          __html: generatedDocs
                            .replace(/\n/g, '<br>')
                            .replace(/```(\w*)\n([\s\S]*?)\n```/g, 
                              '<pre class="bg-gray-200 dark:bg-gray-800 p-2 rounded"><code>$2</code></pre>')
                            .replace(/`([^`]+)`/g, '<code class="bg-gray-200 dark:bg-gray-800 px-1 rounded">$1</code>')
                            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
                            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
                            .replace(/\|([^|]+)\|/g, '<table class="border-collapse border border-gray-300 w-full"><tr><td class="border border-gray-300 p-2">$1</td></tr></table>')
                        }} 
                      />
                    ) : (
                      <pre className="whitespace-pre-wrap font-mono text-sm">
                        {generatedDocs}
                      </pre>
                    )}
                  </div>
                </TabsContent>
                
                <TabsContent value="code">
                  <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-md border">
                    <pre className="whitespace-pre-wrap font-mono text-sm">
                      {generatedDocs}
                    </pre>
                  </div>
                </TabsContent>
              </div>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
