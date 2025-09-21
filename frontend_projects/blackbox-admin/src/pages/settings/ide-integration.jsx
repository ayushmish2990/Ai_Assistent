import React from 'react';
import { useTranslation } from 'react-i18next';
import { IDESettings } from '@/components/settings/IDESettings';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function IDEIntegrationPage() {
  const { t } = useTranslation();
  
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">{t('ideSettings.title')}</h2>
        <p className="text-muted-foreground">
          {t('ideSettings.subtitle')}
        </p>
      </div>
      
      <IDESettings />
      
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>{t('ideSettings.setupInstructions.title')}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Visual Studio Code</h3>
            <ol className="list-decimal list-inside space-y-2 pl-2">
              {t('ideSettings.setupInstructions.vscode.steps', { returnObjects: true }).map((step, i) => (
                <li key={i} className="text-sm text-muted-foreground">
                  {step}
                </li>
              ))}
            </ol>
          </div>
          
          <div className="space-y-4 pt-4 border-t">
            <h3 className="text-lg font-medium">JetBrains IDEs</h3>
            <ol className="list-decimal list-inside space-y-2 pl-2">
              {t('ideSettings.setupInstructions.jetbrains.steps', { returnObjects: true }).map((step, i) => (
                <li key={i} className="text-sm text-muted-foreground">
                  {step}
                </li>
              ))}
            </ol>
          </div>
          
          <div className="space-y-4 pt-4 border-t">
            <h3 className="text-lg font-medium">Vim/Neovim</h3>
            <ol className="list-decimal list-inside space-y-2 pl-2">
              {t('ideSettings.setupInstructions.vim.steps', { returnObjects: true }).map((step, i) => (
                <li key={i} className="text-sm text-muted-foreground">
                  {step}
                </li>
              ))}
            </ol>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
