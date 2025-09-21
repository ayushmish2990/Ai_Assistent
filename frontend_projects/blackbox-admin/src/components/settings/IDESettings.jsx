import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from '@/components/ui/use-toast';

export function IDESettings() {
  const { t } = useTranslation();
  const [settings, setSettings] = useState({
    vscodeEnabled: false,
    jetbrainsEnabled: false,
    vimEnabled: false,
    autoSync: true,
    syncInterval: 5,
    defaultBranch: 'main'
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    // Simulate loading settings
    const loadSettings = async () => {
      try {
        // TODO: Replace with actual API call
        await new Promise(resolve => setTimeout(resolve, 500));
        // Mock data
        const mockSettings = {
          vscodeEnabled: true,
          jetbrainsEnabled: false,
          vimEnabled: false,
          autoSync: true,
          syncInterval: 5,
          defaultBranch: 'main'
        };
        setSettings(mockSettings);
      } catch (error) {
        console.error('Failed to load IDE settings:', error);
        toast({
          title: t('errors.loadingFailed'),
          description: t('ideSettings.errors.loadFailed'),
          variant: 'destructive',
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadSettings();
  }, [t]);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      // TODO: Replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      toast({
        title: t('common.saved'),
        description: t('ideSettings.savedSuccessfully'),
      });
    } catch (error) {
      console.error('Failed to save IDE settings:', error);
      toast({
        title: t('errors.saveFailed'),
        description: t('ideSettings.errors.saveFailed'),
        variant: 'destructive',
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p>{t('common.loading')}...</p>
      </div>
    );
  }

  return (
    <Card className="border-0 shadow-sm">
      <CardHeader>
        <CardTitle>{t('ideSettings.title')}</CardTitle>
        <p className="text-sm text-muted-foreground">
          {t('ideSettings.subtitle')}
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-4">
          <h3 className="text-lg font-medium">{t('ideSettings.integrations')}</h3>
          
          <div className="flex items-center justify-between p-4 border rounded-lg">
            <div>
              <h4 className="font-medium">Visual Studio Code</h4>
              <p className="text-sm text-muted-foreground">
                {t('ideSettings.vscodeDescription')}
              </p>
            </div>
            <Switch
              checked={settings.vscodeEnabled}
              onCheckedChange={(checked) => handleChange('vscodeEnabled', checked)}
            />
          </div>

          <div className="flex items-center justify-between p-4 border rounded-lg">
            <div>
              <h4 className="font-medium">JetBrains IDEs</h4>
              <p className="text-sm text-muted-foreground">
                {t('ideSettings.jetbrainsDescription')}
              </p>
            </div>
            <Switch
              checked={settings.jetbrainsEnabled}
              onCheckedChange={(checked) => handleChange('jetbrainsEnabled', checked)}
            />
          </div>

          <div className="flex items-center justify-between p-4 border rounded-lg">
            <div>
              <h4 className="font-medium">Vim/Neovim</h4>
              <p className="text-sm text-muted-foreground">
                {t('ideSettings.vimDescription')}
              </p>
            </div>
            <Switch
              checked={settings.vimEnabled}
              onCheckedChange={(checked) => handleChange('vimEnabled', checked)}
            />
          </div>
        </div>

        <div className="space-y-4 pt-4 border-t">
          <h3 className="text-lg font-medium">{t('ideSettings.syncSettings')}</h3>
          
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="auto-sync">
                {t('ideSettings.autoSync')}
              </Label>
              <div className="flex items-center space-x-2">
                <Switch
                  id="auto-sync"
                  checked={settings.autoSync}
                  onCheckedChange={(checked) => handleChange('autoSync', checked)}
                />
                <span className="text-sm text-muted-foreground">
                  {settings.autoSync ? t('common.enabled') : t('common.disabled')}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                {t('ideSettings.autoSyncDescription')}
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="sync-interval">
                {t('ideSettings.syncInterval')}
              </Label>
              <Select
                value={settings.syncInterval.toString()}
                onValueChange={(value) => handleChange('syncInterval', parseInt(value))}
                disabled={!settings.autoSync}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select interval" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1 {t('common.minute')}</SelectItem>
                  <SelectItem value="5">5 {t('common.minutes')}</SelectItem>
                  <SelectItem value="15">15 {t('common.minutes')}</SelectItem>
                  <SelectItem value="30">30 {t('common.minutes')}</SelectItem>
                  <SelectItem value="60">1 {t('common.hour')}</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {t('ideSettings.syncIntervalDescription')}
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="default-branch">
              {t('ideSettings.defaultBranch')}
            </Label>
            <Input
              id="default-branch"
              value={settings.defaultBranch}
              onChange={(e) => handleChange('defaultBranch', e.target.value)}
              className="max-w-xs"
            />
            <p className="text-xs text-muted-foreground">
              {t('ideSettings.defaultBranchDescription')}
            </p>
          </div>
        </div>

        <div className="flex justify-end pt-4">
          <Button onClick={handleSave} disabled={isSaving}>
            {isSaving ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t('common.saving')}...
              </>
            ) : (
              t('common.saveChanges')
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
