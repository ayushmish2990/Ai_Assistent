import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Loader2, LayoutDashboard, Database, Cpu, Server, Code2 } from 'lucide-react';
import { useToast } from '../ui/use-toast';

export function ProjectArchitect() {
  const { t } = useTranslation();
  const { toast } = useToast();
  const [projectDescription, setProjectDescription] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('architecture');
  const [results, setResults] = useState({
    architecture: '',
    database: '',
    api: '',
    deployment: ''
  });

  const generatePlan = async (type) => {
    if (!projectDescription.trim()) {
      toast({
        title: t('errors.requiredField'),
        description: t('projectArchitect.errors.enterDescription'),
        variant: 'destructive'
      });
      return;
    }

    setIsLoading(true);
    setActiveTab(type);

    try {
      // TODO: Replace with actual API call to your backend
      // This is a mock implementation
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock response based on type
      const mockResponses = {
        architecture: `# ${t('projectArchitect.results.architectureTitle')}

## ${t('projectArchitect.results.overview')}
${t('projectArchitect.mock.architecture.overview')}

## ${t('projectArchitect.results.components')}
1. **Frontend**: React 18 with TypeScript, Vite, Tailwind CSS
2. **Backend**: Node.js with Express, TypeScript
3. **Database**: PostgreSQL with Prisma ORM
4. **Authentication**: JWT with refresh tokens
5. **API**: RESTful design with OpenAPI documentation

## ${t('projectArchitect.results.dataFlow')}
${t('projectArchitect.mock.architecture.dataFlow')}

## ${t('projectArchitect.results.techStack')}
- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **State Management**: React Query
- **Backend**: Node.js, Express, TypeScript
- **Database**: PostgreSQL, Prisma
- **Auth**: JWT, bcrypt
- **Testing**: Jest, React Testing Library
- **CI/CD**: GitHub Actions, Docker, AWS ECS`,

        database: `# ${t('projectArchitect.results.databaseTitle')}

## ${t('projectArchitect.results.schema')}

### users
- id: UUID (Primary Key)
- email: String (Unique, Indexed)
- password_hash: String
- name: String
- role: Enum('user', 'admin')
- created_at: Timestamp
- updated_at: Timestamp

### projects
- id: UUID (Primary Key)
- name: String
- description: Text
- owner_id: UUID (Foreign Key to users)
- status: Enum('planning', 'in_progress', 'completed', 'on_hold')
- created_at: Timestamp
- updated_at: Timestamp

### tasks
- id: UUID (Primary Key)
- project_id: UUID (Foreign Key to projects)
- title: String
- description: Text
- status: Enum('todo', 'in_progress', 'in_review', 'done')
- assignee_id: UUID (Foreign Key to users, Nullable)
- due_date: Timestamp (Nullable)
- created_at: Timestamp
- updated_at: Timestamp

## ${t('projectArchitect.results.relationships')}
- One User can have many Projects (One-to-Many)
- One Project can have many Tasks (One-to-Many)
- One User can be assigned to many Tasks (One-to-Many)

## ${t('projectArchitect.results.indexes')}
- Index on users.email for fast lookups
- Index on projects.owner_id for user-specific queries
- Index on tasks.project_id for project-specific queries
- Index on tasks.assignee_id for user assignment lookups
- Index on tasks.status for filtering tasks by status`,

        api: `# ${t('projectArchitect.results.apiTitle')}

## ${t('projectArchitect.results.authentication')}

### POST /api/auth/register
- ${t('projectArchitect.mock.api.register')}
- **Request Body**:
  \`\`\`json
  {
    "email": "user@example.com",
    "password": "securepassword123",
    "name": "John Doe"
  }
  \`\`\`

### POST /api/auth/login
- ${t('projectArchitect.mock.api.login')}
- **Request Body**:
  \`\`\`json
  {
    "email": "user@example.com",
    "password": "securepassword123"
  }
  \`\`\`

## ${t('projectArchitect.results.projects')}

### GET /api/projects
- ${t('projectArchitect.mock.api.listProjects')}
- **Headers**: \`Authorization: Bearer <token>\`
- **Query Params**:
  - \`status\`: Filter by status
  - \`sort\`: Sort field (created_at, updated_at, name)
  - \`order\`: Sort order (asc, desc)

### POST /api/projects
- ${t('projectArchitect.mock.api.createProject')}
- **Headers**: \`Authorization: Bearer <token>\`
- **Request Body**:
  \`\`\`json
  {
    "name": "New Project",
    "description": "Project description"
  }
  \`\`\``
      };

      setResults(prev => ({
        ...prev,
        [type]: mockResponses[type] || t('projectArchitect.errors.noData')
      }));

    } catch (error) {
      console.error('Error generating plan:', error);
      toast({
        title: t('errors.error'),
        description: t('projectArchitect.errors.generationFailed'),
        variant: 'destructive'
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle>{t('projectArchitect.projectDetails')}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label htmlFor="project-description" className="block text-sm font-medium mb-1">
                {t('projectArchitect.labels.projectDescription')}
              </label>
              <Textarea
                id="project-description"
                value={projectDescription}
                onChange={(e) => setProjectDescription(e.target.value)}
                placeholder={t('projectArchitect.placeholders.projectDescription')}
                className="min-h-[150px]"
                disabled={isLoading}
              />
            </div>

            <div className="flex flex-wrap gap-2">
              <Button 
                onClick={() => generatePlan('architecture')}
                disabled={isLoading || !projectDescription.trim()}
                variant="outline"
                className="gap-2"
              >
                {isLoading && activeTab === 'architecture' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <LayoutDashboard className="h-4 w-4" />
                )}
                {t('projectArchitect.actions.generateArchitecture')}
              </Button>
              
              <Button 
                onClick={() => generatePlan('database')}
                disabled={isLoading || !projectDescription.trim()}
                variant="outline"
                className="gap-2"
              >
                {isLoading && activeTab === 'database' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Database className="h-4 w-4" />
                )}
                {t('projectArchitect.actions.generateDatabase')}
              </Button>
              
              <Button 
                onClick={() => generatePlan('api')}
                disabled={isLoading || !projectDescription.trim()}
                variant="outline"
                className="gap-2"
              >
                {isLoading && activeTab === 'api' ? (
                  <Cpu className="h-4 w-4 animate-spin" />
                ) : (
                  <Code2 className="h-4 w-4" />
                )}
                {t('projectArchitect.actions.generateApi')}
              </Button>
              
              <Button 
                onClick={() => generatePlan('deployment')}
                disabled={isLoading || !projectDescription.trim()}
                variant="outline"
                className="gap-2"
              >
                {isLoading && activeTab === 'deployment' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Server className="h-4 w-4" />
                )}
                {t('projectArchitect.actions.generateDeployment')}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {(results.architecture || results.database || results.api || results.deployment) && (
        <Card className="border-0 shadow-sm">
          <CardHeader>
            <CardTitle>{t('projectArchitect.results.title')}</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="architecture" disabled={!results.architecture}>
                  <LayoutDashboard className="mr-2 h-4 w-4" />
                  {t('projectArchitect.tabs.architecture')}
                </TabsTrigger>
                <TabsTrigger value="database" disabled={!results.database}>
                  <Database className="mr-2 h-4 w-4" />
                  {t('projectArchitect.tabs.database')}
                </TabsTrigger>
                <TabsTrigger value="api" disabled={!results.api}>
                  <Cpu className="mr-2 h-4 w-4" />
                  {t('projectArchitect.tabs.api')}
                </TabsTrigger>
                <TabsTrigger value="deployment" disabled={!results.deployment}>
                  <Server className="mr-2 h-4 w-4" />
                  {t('projectArchitect.tabs.deployment')}
                </TabsTrigger>
              </TabsList>
              
              <div className="mt-6">
                <TabsContent value="architecture" className="space-y-4">
                  <div className="prose dark:prose-invert max-w-none">
                    <pre className="whitespace-pre-wrap">{results.architecture}</pre>
                  </div>
                </TabsContent>
                
                <TabsContent value="database" className="space-y-4">
                  <div className="prose dark:prose-invert max-w-none">
                    <pre className="whitespace-pre-wrap">{results.database}</pre>
                  </div>
                </TabsContent>
                
                <TabsContent value="api" className="space-y-4">
                  <div className="prose dark:prose-invert max-w-none">
                    <pre className="whitespace-pre-wrap">{results.api}</pre>
                  </div>
                </TabsContent>
                
                <TabsContent value="deployment" className="space-y-4">
                  <div className="prose dark:prose-invert max-w-none">
                    <pre className="whitespace-pre-wrap">{results.deployment}</pre>
                  </div>
                </TabsContent>
              </div>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
