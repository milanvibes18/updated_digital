import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { login } from '@/utils/api';
import { useToast } from '@/hooks/use-toast';
import { Loader2, Wifi, WifiOff } from 'lucide-react';

// Health check endpoint from your backend API
const API_HEALTH_URL = `${import.meta.env.VITE_API_URL}/health`;

export function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(localStorage.getItem('demo_mode') === 'true');
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const navigate = useNavigate();
  const { toast } = useToast();

  // Check Backend Status
  useEffect(() => {
    const checkStatus = async () => {
      if (isDemoMode) {
        setBackendStatus('offline');
        return;
      }
      try {
        const response = await fetch(API_HEALTH_URL);
        setBackendStatus(response.ok ? 'online' : 'offline');
      } catch (error) {
        setBackendStatus('offline');
        console.error("Backend health check failed:", error);
      }
    };

    checkStatus();
    const intervalId = setInterval(checkStatus, 30000);
    return () => clearInterval(intervalId);
  }, [isDemoMode]);

  // Handle Demo Mode Toggle
  const handleDemoModeChange = (checked: boolean) => {
    setIsDemoMode(checked);
    if (checked) {
      localStorage.setItem('demo_mode', 'true');
      toast({
        title: "Demo Mode Enabled",
        description: "The application will now use local sample data.",
      });
    } else {
      localStorage.removeItem('demo_mode');
      setBackendStatus('checking');
      toast({
        title: "Demo Mode Disabled",
        description: "Connecting to the live backend API.",
      });
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsLoading(true);

    if (isDemoMode) {
      setTimeout(() => {
        setIsLoading(false);
        navigate('/dashboard');
      }, 500);
      return;
    }

    try {
      if (backendStatus !== 'online') {
         throw new Error("Backend is offline. Please try again later.");
      }
      const data = await login(username, password);
      localStorage.setItem('jwt_token', data.token);
      localStorage.setItem('user_id', data.userId);
      navigate('/dashboard');
    } catch (error) {
      toast({
        title: 'Login Failed',
        description: error instanceof Error ? error.message : 'An unknown error occurred.',
        variant: 'destructive',
      });
    } finally {
        setIsLoading(false);
    }
  };

  const getStatusIndicator = () => {
      if (isDemoMode) {
          return <span className="text-blue-500">Demo Mode</span>;
      }
      switch (backendStatus) {
          case 'online':
              return <><Wifi className="h-4 w-4 text-green-500" /> <span className="text-green-500">Online</span></>;
          case 'offline':
              return <><WifiOff className="h-4 w-4 text-red-500" /> <span className="text-red-500">Offline</span></>;
          default:
              return <><Loader2 className="h-4 w-4 animate-spin" /> <span>Checking...</span></>;
      }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-background">
      <Card className="w-full max-w-md mx-4 shadow-xl">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-2xl">Login</CardTitle>
            <div className="flex items-center gap-2 text-xs font-medium">
                {getStatusIndicator()}
            </div>
          </div>
          <CardDescription>
            Enter your credentials or use Demo Mode to continue.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                type="text"
                placeholder="Enter your username"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={isLoading || isDemoMode}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                placeholder="Enter your password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isLoading || isDemoMode}
              />
            </div>
            <div className="flex items-center justify-between pt-2 border-t mt-4">
              <Label htmlFor="demo-mode" className="flex flex-col gap-1 cursor-pointer">
                <span className="font-semibold">Demo Mode</span>
                 <span className="text-xs font-normal text-muted-foreground">
                    Use local data without logging in.
                </span>
              </Label>
              <Switch
                id="demo-mode"
                checked={isDemoMode}
                onCheckedChange={handleDemoModeChange}
                disabled={isLoading}
              />
            </div>
            <Button type="submit" className="w-full" disabled={isLoading || (backendStatus !== 'online' && !isDemoMode)}>
              {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              {isDemoMode ? 'Enter Demo' : 'Login'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

