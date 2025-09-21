import { Outlet } from 'react-router-dom';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Menu, X, LayoutDashboard, Users, Bug, Settings, LayoutGrid, Code, FileText, TestTube2, Code2 } from 'lucide-react';
import { cn } from '../../lib/utils';
import { Button } from '../ui/button';
import { useTheme } from 'next-themes';
import { Moon, Sun } from 'lucide-react';
import { LanguageSelector } from '../ui/language-selector';

const navigation = (t) => [
  { name: t('navigation.dashboard'), href: '/', icon: 'LayoutDashboard' },
  { name: t('navigation.users'), href: '/users', icon: 'Users' },
  { name: t('navigation.projectArchitect'), href: '/project-architect', icon: 'LayoutGrid' },
  { name: t('navigation.codeReview'), href: '/code-review', icon: 'Code' },
  { name: t('navigation.documentation'), href: '/documentation', icon: 'FileText' },
  { name: t('navigation.testGenerator'), href: '/test-generator', icon: 'TestTube2' },
  { name: t('navigation.ideIntegration'), href: '/ide-integration', icon: 'Code2' },
  { name: t('navigation.debug'), href: '/debug', icon: 'Bug' },
  { name: t('navigation.settings'), href: '/settings', icon: 'Settings' },
];

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { theme, setTheme } = useTheme();
  const { t } = useTranslation();
  
  // Use the same navigation items for both mobile and desktop
  const navItems = navigation(t);

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile sidebar */}
      <div className="lg:hidden">
        <div className="fixed inset-0 z-40 flex">
          <div
            className={cn(
              'relative flex w-full max-w-xs flex-1 flex-col bg-background pt-5 pb-4',
              {
                'translate-x-0': sidebarOpen,
                '-translate-x-full': !sidebarOpen,
              },
              'transition-transform duration-200 ease-in-out'
            )}
          >
            <div className="absolute top-0 right-0 -mr-12 pt-2">
              <button
                type="button"
                className="ml-1 flex h-10 w-10 items-center justify-center rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
                onClick={() => setSidebarOpen(false)}
              >
                <span className="sr-only">Close sidebar</span>
                <X className="h-6 w-6 text-white" aria-hidden="true" />
              </button>
            </div>
            <div className="flex flex-shrink-0 items-center px-4">
              <h1 className="text-xl font-bold">Blackbox AI</h1>
            </div>
            <nav
              className="mt-5 h-full flex-shrink-0 divide-y divide-gray-200 overflow-y-auto"
              aria-label="Sidebar"
            >
              <div className="space-y-1 px-2">
                {navItems.map((item) => {
                  const Icon = item.icon === 'LayoutDashboard' ? LayoutDashboard :
                             item.icon === 'Users' ? Users :
                             item.icon === 'LayoutGrid' ? LayoutGrid :
                             item.icon === 'Code' ? Code :
                             item.icon === 'FileText' ? FileText :
                             item.icon === 'TestTube2' ? TestTube2 :
                             item.icon === 'Code2' ? Code2 :
                             item.icon === 'Bug' ? Bug :
                             item.icon === 'Settings' ? Settings : null;
                             
                  return (
                    <a
                      key={item.href}
                      href={item.href}
                      className="group flex items-center rounded-md px-2 py-2 text-base font-medium text-foreground hover:bg-accent hover:text-accent-foreground"
                    >
                      {Icon && <Icon className="mr-4 h-6 w-6 flex-shrink-0" />}
                      {item.name}
                    </a>
                  );
                })}
              </div>
            </nav>
          </div>
          <div className="w-14 flex-shrink-0" aria-hidden="true">
            {/* Dummy element to force sidebar to shrink to fit close icon */}
          </div>
        </div>
      </div>

      {/* Static sidebar for desktop */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col">
        <div className="flex min-h-0 flex-1 flex-col border-r bg-background">
          <div className="flex flex-1 flex-col overflow-y-auto pt-5 pb-4">
            <div className="flex flex-shrink-0 items-center px-4">
              <h1 className="text-xl font-bold">Blackbox AI</h1>
            </div>
            <nav
              className="mt-5 flex-1 space-y-1 px-2"
              aria-label="Sidebar"
            >
              {navigation(t).map((item) => {
                const Icon = item.icon === 'LayoutDashboard' ? LayoutDashboard :
                           item.icon === 'Users' ? Users :
                           item.icon === 'LayoutGrid' ? LayoutGrid :
                           item.icon === 'Code' ? Code :
                           item.icon === 'FileText' ? FileText :
                           item.icon === 'TestTube2' ? TestTube2 :
                           item.icon === 'Code2' ? Code2 :
                           item.icon === 'Bug' ? Bug :
                           item.icon === 'Settings' ? Settings : null;
                           
                return (
                  <a
                    key={item.href}
                    href={item.href}
                    className="group flex items-center rounded-md px-2 py-2 text-sm font-medium text-foreground hover:bg-accent hover:text-accent-foreground"
                  >
                    {Icon && <Icon className="mr-3 h-6 w-6 flex-shrink-0" />}
                    {item.name}
                  </a>
                );
              })}
            </nav>
          </div>
          <div className="flex flex-shrink-0 border-t p-4">
            <div className="group block w-full flex-shrink-0">
              <div className="flex items-center">
                <div>
                  <div className="flex items-center space-x-2">
                    <LanguageSelector />
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
                    >
                      {theme === 'light' ? (
                        <Moon className="h-4 w-4" />
                      ) : (
                        <Sun className="h-4 w-4" />
                      )}
                      <span className="sr-only">Toggle theme</span>
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-col lg:pl-64">
        <div className="sticky top-0 z-10 flex h-16 flex-shrink-0 bg-background shadow">
          <button
            type="button"
            className="border-r border-gray-200 px-4 text-gray-500 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500 lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <span className="sr-only">Open sidebar</span>
            <Menu className="h-6 w-6" aria-hidden="true" />
          </button>
          <div className="flex flex-1 justify-between px-4">
            <div className="flex flex-1">
              {/* Search bar can go here */}
            </div>
            <div className="ml-4 flex items-center lg:ml-6">
              {/* Profile dropdown */}
              <div className="relative">
                <div>
                  <button
                    type="button"
                    className="flex max-w-xs items-center rounded-full bg-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
                    id="user-menu-button"
                    aria-expanded="false"
                    aria-haspopup="true"
                  >
                    <span className="sr-only">Open user menu</span>
                    <div className="h-8 w-8 rounded-full bg-indigo-600 flex items-center justify-center text-white font-medium">
                      U
                    </div>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <main className="flex-1">
          <div className="py-6">
            <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
              <Outlet />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
