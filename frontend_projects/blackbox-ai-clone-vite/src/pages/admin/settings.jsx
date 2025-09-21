import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Settings, AlertCircle, Users, Lock, Mail, Server, CreditCard, Bell, Shield } from 'lucide-react';

export default function SystemSettings() {
  const [isSaving, setIsSaving] = useState(false);
  const [settings, setSettings] = useState({
    siteName: 'Blackbox AI',
    siteDescription: 'Advanced AI Assistant Platform',
    maintenanceMode: false,
    userRegistration: true,
    emailNotifications: true,
    maxUsers: 1000,
    storageLimit: 100, // in GB
    currentPlan: 'enterprise',
    apiEnabled: true,
    apiRateLimit: 1000,
  });

  const handleSave = () => {
    setIsSaving(true);
    // Simulate API call
    setTimeout(() => {
      setIsSaving(false);
      // Show success message
    }, 1500);
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setSettings(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">System Settings</h2>
          <p className="text-sm text-muted-foreground">
            Configure system-wide settings and preferences
          </p>
        </div>
        <Button onClick={handleSave} disabled={isSaving}>
          {isSaving ? 'Saving...' : 'Save Changes'}
        </Button>
      </div>

      <Tabs defaultValue="general" className="space-y-4">
        <TabsList>
          <TabsTrigger value="general">
            <Settings className="w-4 h-4 mr-2" />
            General
          </TabsTrigger>
          <TabsTrigger value="users">
            <Users className="w-4 h-4 mr-2" />
            Users
          </TabsTrigger>
          <TabsTrigger value="security">
            <Shield className="w-4 h-4 mr-2" />
            Security
          </TabsTrigger>
          <TabsTrigger value="notifications">
            <Bell className="w-4 h-4 mr-2" />
            Notifications
          </TabsTrigger>
          <TabsTrigger value="billing">
            <CreditCard className="w-4 h-4 mr-2" />
            Billing
          </TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>General Settings</CardTitle>
              <CardDescription>Configure your application settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="siteName">Site Name</Label>
                  <Input 
                    id="siteName" 
                    name="siteName" 
                    value={settings.siteName}
                    onChange={handleInputChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="siteDescription">Site Description</Label>
                  <Input 
                    id="siteDescription" 
                    name="siteDescription" 
                    value={settings.siteDescription}
                    onChange={handleInputChange}
                  />
                </div>
              </div>
              
              <div className="flex items-center justify-between rounded-lg border p-4">
                <div className="space-y-0.5">
                  <Label htmlFor="maintenance-mode">Maintenance Mode</Label>
                  <p className="text-sm text-muted-foreground">
                    When enabled, the site will be in maintenance mode and only accessible to admins.
                  </p>
                </div>
                <Switch 
                  id="maintenance-mode" 
                  checked={settings.maintenanceMode}
                  onCheckedChange={(checked) => setSettings({...settings, maintenanceMode: checked})}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="users" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>User Settings</CardTitle>
              <CardDescription>Manage user registration and permissions</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between rounded-lg border p-4">
                <div className="space-y-0.5">
                  <Label htmlFor="user-registration">Allow New User Registration</Label>
                  <p className="text-sm text-muted-foreground">
                    Allow new users to create accounts on your platform.
                  </p>
                </div>
                <Switch 
                  id="user-registration" 
                  checked={settings.userRegistration}
                  onCheckedChange={(checked) => setSettings({...settings, userRegistration: checked})}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="maxUsers">Maximum Users</Label>
                <Input 
                  id="maxUsers" 
                  name="maxUsers" 
                  type="number"
                  value={settings.maxUsers}
                  onChange={handleInputChange}
                  className="w-32"
                />
                <p className="text-sm text-muted-foreground">
                  Maximum number of users allowed on your plan.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Security Settings</CardTitle>
              <CardDescription>Configure security preferences</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div className="space-y-0.5">
                    <Label htmlFor="api-enabled">Enable API Access</Label>
                    <p className="text-sm text-muted-foreground">
                      Allow users to access the API with their API keys.
                    </p>
                  </div>
                  <Switch 
                    id="api-enabled" 
                    checked={settings.apiEnabled}
                    onCheckedChange={(checked) => setSettings({...settings, apiEnabled: checked})}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="apiRateLimit">API Rate Limit (per hour)</Label>
                  <div className="flex items-center space-x-2">
                    <Input 
                      id="apiRateLimit" 
                      name="apiRateLimit" 
                      type="number"
                      value={settings.apiRateLimit}
                      onChange={handleInputChange}
                      className="w-32"
                      disabled={!settings.apiEnabled}
                    />
                    <span className="text-sm text-muted-foreground">requests/hour</span>
                  </div>
                </div>
                
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Security Notice</AlertTitle>
                  <AlertDescription>
                    Changing these settings may affect the security of your application. Please ensure you understand the implications before making changes.
                  </AlertDescription>
                </Alert>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Notification Settings</CardTitle>
              <CardDescription>Configure how you receive notifications</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div className="space-y-0.5">
                    <Label htmlFor="email-notifications">Email Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Receive email notifications for important updates.
                    </p>
                  </div>
                  <Switch 
                    id="email-notifications" 
                    checked={settings.emailNotifications}
                    onCheckedChange={(checked) => setSettings({...settings, emailNotifications: checked})}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="billing" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Billing & Plan</CardTitle>
              <CardDescription>Manage your subscription and billing information</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <div className="flex items-center space-x-2">
                      <h3 className="font-medium">Current Plan: {settings.currentPlan}</h3>
                      <Badge variant="secondary" className="uppercase">
                        {settings.currentPlan}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                      {settings.storageLimit}GB Storage • Unlimited Users • Priority Support
                    </p>
                  </div>
                  <Button variant="outline">Upgrade Plan</Button>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium mb-2">Basic</h4>
                    <p className="text-2xl font-bold mb-2">$29<small className="text-sm font-normal text-muted-foreground">/month</small></p>
                    <p className="text-sm text-muted-foreground mb-4">For individuals getting started</p>
                    <Button variant="outline" className="w-full">Select Plan</Button>
                  </div>
                  <div className="border-2 border-primary rounded-lg p-4 relative">
                    <div className="absolute -top-2 right-4 bg-primary text-primary-foreground px-2 py-0.5 rounded-full text-xs font-medium">
                      Current
                    </div>
                    <h4 className="font-medium mb-2">Pro</h4>
                    <p className="text-2xl font-bold mb-2">$99<small className="text-sm font-normal text-muted-foreground">/month</small></p>
                    <p className="text-sm text-muted-foreground mb-4">For growing teams</p>
                    <Button className="w-full">Current Plan</Button>
                  </div>
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium mb-2">Enterprise</h4>
                    <p className="text-2xl font-bold mb-2">Custom</p>
                    <p className="text-sm text-muted-foreground mb-4">For large organizations</p>
                    <Button variant="outline" className="w-full">Contact Sales</Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
