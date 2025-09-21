import React from 'react';
import { useTranslation } from 'react-i18next';
import { ProjectArchitect } from '@/components/ai/ProjectArchitect';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function ProjectArchitectPage() {
  const { t } = useTranslation();
  
  return (
    <div className="container mx-auto py-8 px-4">
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>{t('projectArchitect.title')}</CardTitle>
          <p className="text-sm text-muted-foreground">
            {t('projectArchitect.subtitle')}
          </p>
        </CardHeader>
        <CardContent>
          <ProjectArchitect />
        </CardContent>
      </Card>
    </div>
  );
}
