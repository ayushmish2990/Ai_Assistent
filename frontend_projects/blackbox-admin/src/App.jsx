import { Routes, Route } from 'react-router-dom';
import { ThemeProvider } from 'next-themes';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { I18nextProvider } from 'react-i18next';
import i18n from './lib/i18n';
import { Toaster } from './components/ui/toaster';
import Layout from './components/layout';
import Dashboard from './pages/dashboard';
import Users from './pages/users';
import Debug from './pages/debug';
import ProjectArchitect from './pages/project-architect';
import CodeReview from './pages/code-review';
import DocumentationGenerator from './pages/documentation-generator';
import TestGenerator from './pages/test-generator';
import IDEIntegration from './pages/settings/ide-integration';
import Settings from './pages/settings';
import NotFound from './pages/not-found';
import { useEffect } from 'react';

const queryClient = new QueryClient();

function App() {
  // Initialize language from localStorage or browser language
  useEffect(() => {
    const savedLanguage = localStorage.getItem('userLanguage');
    if (savedLanguage) {
      i18n.changeLanguage(savedLanguage);
    } else {
      const browserLanguage = navigator.language.split('-')[0];
      if (i18n.languages.includes(browserLanguage)) {
        i18n.changeLanguage(browserLanguage);
      }
    }
  }, []);

  return (
    <I18nextProvider i18n={i18n}>
      <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
        <QueryClientProvider client={queryClient}>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="users" element={<Users />} />
              <Route path="project-architect" element={<ProjectArchitect />} />
              <Route path="code-review" element={<CodeReview />} />
              <Route path="documentation" element={<DocumentationGenerator />} />
              <Route path="test-generator" element={<TestGenerator />} />
              <Route path="ide-integration" element={<IDEIntegration />} />
              <Route path="debug" element={<Debug />} />
              <Route path="settings" element={<Settings />} />
              <Route path="*" element={<NotFound />} />
            </Route>
          </Routes>
          <Toaster />
        </QueryClientProvider>
      </ThemeProvider>
    </I18nextProvider>
  );
}

export default App;
