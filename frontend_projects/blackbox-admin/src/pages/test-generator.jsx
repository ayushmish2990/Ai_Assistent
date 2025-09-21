import React from 'react';
import { useTranslation } from 'react-i18next';
import { TestGenerator } from '@/components/ai/TestGenerator';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function TestGeneratorPage() {
  const { t } = useTranslation();
  
  return (
    <div className="container mx-auto py-8 px-4">
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>{t('tests.title')}</CardTitle>
          <p className="text-sm text-muted-foreground">
            {t('tests.subtitle')}
          </p>
        </CardHeader>
        <CardContent>
          <TestGenerator />
        </CardContent>
      </Card>
    </div>
  );
}
