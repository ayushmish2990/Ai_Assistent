import { DebugAssistant } from '../components/ai/DebugAssistant';
import { Card } from '../components/ui/card';

export default function DebugPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">AI Debug Assistant</h1>
        <p className="text-muted-foreground">
          Analyze and fix code issues with the power of AI
        </p>
      </div>
      
      <Card className="p-6">
        <DebugAssistant />
      </Card>
    </div>
  );
}
