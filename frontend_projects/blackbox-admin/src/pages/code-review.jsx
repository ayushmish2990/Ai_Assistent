import React from 'react';
import { useTranslation } from 'react-i18next';
import { CodeReview } from '@/components/ai/CodeReview';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function CodeReviewPage() {
  const { t } = useTranslation();
  
  return (
    <div className="container mx-auto py-8 px-4">
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>{t('codeReview.title')}</CardTitle>
          <p className="text-sm text-muted-foreground">
            {t('codeReview.subtitle')}
          </p>
        </CardHeader>
        <CardContent>
          <CodeReview />
        </CardContent>
      </Card>
    </div>
  );
}
