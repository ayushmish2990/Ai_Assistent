import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Switch } from '../components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';

export default function Settings() {
  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold tracking-tight">Settings</h2>
        <p className="text-muted-foreground">
          Manage your application settings
        </p>
      </div>

      <Tabs defaultValue="general" className="space-y-4">
        <TabsList>
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="users">Users</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="billing">Billing</TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="space-y-4">
          <div className="grid gap-6">
            <div className="space-y-2">
              <h3 className="text-lg font-medium">General Settings</h3>
              <p className="text-sm text-muted-foreground">
                Configure general application settings
              </p>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="app-name">Application Name</Label>
                <Input id="app-name" defaultValue="Blackbox AI" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="timezone">Timezone</Label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a timezone" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="est">Eastern Standard Time (EST)</SelectItem>
                    <SelectItem value="cst">Central Standard Time (CST)</SelectItem>
                    <SelectItem value="mst">Mountain Standard Time (MST)</SelectItem>
                    <SelectItem value="pst">Pacific Standard Time (PST)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="date-format">Date Format</Label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a date format" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mm/dd/yyyy">MM/DD/YYYY</SelectItem>
                    <SelectItem value="dd/mm/yyyy">DD/MM/YYYY</SelectItem>
                    <SelectItem value="yyyy-mm-dd">YYYY-MM-DD</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="language">Language</Label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a language" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="en">English</SelectItem>
                    <SelectItem value="es">Español</SelectItem>
                    <SelectItem value="fr">Français</SelectItem>
                    <SelectItem value="de">Deutsch</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="flex justify-end">
              <Button>Save Changes</Button>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="users" className="space-y-4">
          <div className="grid gap-6">
            <div className="space-y-2">
              <h3 className="text-lg font-medium">User Settings</h3>
              <p className="text-sm text-muted-foreground">
                Configure user management settings
              </p>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between space-x-2">
                <div>
                  <h4 className="font-medium">Allow new user registrations</h4>
                  <p className="text-sm text-muted-foreground">
                    Allow new users to sign up for an account
                  </p>
                </div>
                <Switch id="allow-registrations" />
              </div>
              <div className="flex items-center justify-between space-x-2">
                <div>
                  <h4 className="font-medium">Require email verification</h4>
                  <p className="text-sm text-muted-foreground">
                    Users must verify their email address before accessing the application
                  </p>
                </div>
                <Switch id="email-verification" defaultChecked />
              </div>
              <div className="flex items-center justify-between space-x-2">
                <div>
                  <h4 className="font-medium">Enable two-factor authentication</h4>
                  <p className="text-sm text-muted-foreground">
                    Require users to enable two-factor authentication
                  </p>
                </div>
                <Switch id="two-factor-auth" />
              </div>
            </div>
            <div className="flex justify-end">
              <Button>Save Changes</Button>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <div className="grid gap-6">
            <div className="space-y-2">
              <h3 className="text-lg font-medium">Security Settings</h3>
              <p className="text-sm text-muted-foreground">
                Configure security and access control settings
              </p>
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="session-timeout">Session Timeout</Label>
                <Select defaultValue="30">
                  <SelectTrigger>
                    <SelectValue placeholder="Select session timeout" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="15">15 minutes</SelectItem>
                    <SelectItem value="30">30 minutes</SelectItem>
                    <SelectItem value="60">1 hour</SelectItem>
                    <SelectItem value="120">2 hours</SelectItem>
                    <SelectItem value="0">No timeout</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-sm text-muted-foreground">
                  How long before an idle user is automatically logged out
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="password-policy">Password Policy</Label>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Switch id="require-uppercase" defaultChecked />
                    <Label htmlFor="require-uppercase">Require uppercase letters</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch id="require-numbers" defaultChecked />
                    <Label htmlFor="require-numbers">Require numbers</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch id="require-special-chars" defaultChecked />
                    <Label htmlFor="require-special-chars">Require special characters</Label>
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="password-expiry">Password Expiry</Label>
                <Select defaultValue="90">
                  <SelectTrigger>
                    <SelectValue placeholder="Select password expiry period" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="30">30 days</SelectItem>
                    <SelectItem value="60">60 days</SelectItem>
                    <SelectItem value="90">90 days</SelectItem>
                    <SelectItem value="180">6 months</SelectItem>
                    <SelectItem value="365">1 year</SelectItem>
                    <SelectItem value="0">Never expire</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-sm text-muted-foreground">
                  How often users must change their password
                </p>
              </div>
            </div>
            <div className="flex justify-end">
              <Button>Save Changes</Button>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-4">
          <div className="grid gap-6">
            <div className="space-y-2">
              <h3 className="text-lg font-medium">Notification Settings</h3>
              <p className="text-sm text-muted-foreground">
                Configure how you receive notifications
              </p>
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <h4 className="font-medium">Email Notifications</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="email-product-updates">Product updates</Label>
                    <Switch id="email-product-updates" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label htmlFor="email-security-alerts">Security alerts</Label>
                    <Switch id="email-security-alerts" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label htmlFor="email-newsletter">Newsletter</Label>
                    <Switch id="email-newsletter" defaultChecked />
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium">In-app Notifications</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="in-app-updates">System updates</Label>
                    <Switch id="in-app-updates" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label htmlFor="in-app-reminders">Reminders</Label>
                    <Switch id="in-app-reminders" defaultChecked />
                  </div>
                </div>
              </div>
            </div>
            <div className="flex justify-end">
              <Button>Save Changes</Button>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="billing" className="space-y-4">
          <div className="grid gap-6">
            <div className="space-y-2">
              <h3 className="text-lg font-medium">Billing Information</h3>
              <p className="text-sm text-muted-foreground">
                Manage your subscription and billing details
              </p>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="card-number">Card Number</Label>
                <Input id="card-number" placeholder="•••• •••• •••• ••••" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="card-holder">Card Holder</Label>
                <Input id="card-holder" placeholder="John Doe" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="expiry-date">Expiry Date</Label>
                <Input id="expiry-date" placeholder="MM/YY" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="cvv">CVV</Label>
                <Input id="cvv" placeholder="•••" />
              </div>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium">Current Plan</h4>
              <div className="rounded-md border p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Pro Plan</p>
                    <p className="text-sm text-muted-foreground">$29/month</p>
                  </div>
                  <Button variant="outline">Change Plan</Button>
                </div>
              </div>
            </div>
            <div className="flex justify-end space-x-2">
              <Button variant="outline">Cancel</Button>
              <Button>Update Payment Method</Button>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
