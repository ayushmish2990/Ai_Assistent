import React from 'react';
import { useTranslation } from 'react-i18next';
import { DocumentationGenerator } from '@/components/ai/DocumentationGenerator';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function DocumentationGeneratorPage() {
  const { t } = useTranslation();
  
  return (
    <div className="container mx-auto py-8 px-4">
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>{t('docs.title')}</CardTitle>
          <p className="text-sm text-muted-foreground">
            {t('docs.subtitle')}
          </p>
        </CardHeader>
        <CardContent>
          <DocumentationGenerator />
        </CardContent>
      </Card>
    </div>
  );
}
