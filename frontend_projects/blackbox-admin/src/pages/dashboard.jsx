import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { BarChart, Users, FileText, MessageSquare } from 'lucide-react';

export default function Dashboard() {
  const { t } = useTranslation();
  
  const stats = [
    { name: t('dashboard.stats.users'), value: '2,345', icon: Users, key: 'users' },
    { name: t('dashboard.stats.queries'), value: '12,345', icon: FileText, key: 'queries' },
    { name: t('dashboard.stats.sessions'), value: '1,234', icon: MessageSquare, key: 'sessions' },
    { name: t('dashboard.stats.requests'), value: '45,678', icon: BarChart, key: 'requests' },
  ];

  return (
    <div>
      <h2 className="text-2xl font-bold tracking-tight">{t('dashboard.title')}</h2>
      <p className="text-muted-foreground">
        {t('dashboard.welcome', { name: 'Admin' })}
      </p>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mt-6">
        {stats.map((stat) => (
          <Card key={stat.name}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {stat.name}
              </CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">
                +20.1% from last month
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7 mt-6">
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>{t('dashboard.overview')}</CardTitle>
          </CardHeader>
          <CardContent className="pl-2">
            <div className="h-[300px] flex items-center justify-center text-muted-foreground">
              {t('dashboard.chartPlaceholder')}
            </div>
          </CardContent>
        </Card>
        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>{t('dashboard.stats.activity')}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="flex items-center space-x-4">
                  <div className="h-2 w-2 rounded-full bg-primary" />
                  <div className="flex-1 space-y-1">
                    <p className="text-sm font-medium leading-none">
                      {t('dashboard.activity.title', { index: i })}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {t('dashboard.activity.description', { index: i })}
                    </p>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {i}{t('dashboard.activity.timeAgo')}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
