// src/pages/Login.tsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Checkbox } from '../components/ui/checkbox'; // Import Checkbox
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert'; // Import Alert
import { loginUser } from '../utils/api';
import { useToast } from '../hooks/use-toast';
import { AlertCircle, Loader2, Wand2 } from 'lucide-react';
import { startSimulation } from '../utils/data-simulator'; // Import simulator

export function Login() {
  const [username, setUsername] = useState('admin'); // Default for convenience
  const [password, setPassword] = useState('password'); // Default for convenience
  const [demoMode, setDemoMode] = useState(false); // --- NEW: Demo mode state ---
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // --- NEW: Handle Demo Mode ---
    if (demoMode) {
      localStorage.setItem('demo_mode', 'true');
      // Ensure any real token is cleared
      localStorage.removeItem('jwt_token'); 
      
      // Start the data simulator to populate IndexedDB
      try {
        await startSimulation(); // Assuming startSimulation is async and populates DB
        toast({
          title: 'Demo Mode Activated',
          description: 'Loading simulated environment...',
        });
        navigate('/dashboard');
      } catch (simError: any) {
        setError(`Failed to start simulator: ${simError.message}`);
        localStorage.removeItem('demo_mode');
        setLoading(false);
      }
      return; // Stop execution
    }

    // --- Standard Login Flow ---
    localStorage.removeItem('demo_mode'); // Ensure demo mode is off

    try {
      // --- UPDATED: No longer storing token in localStorage ---
      // The loginUser function will be called, and the backend
      // is expected to set an HttpOnly cookie in the response.
      await loginUser({ username, password });

      // --- REMOVED: localStorage.setItem('jwt_token', data.access_token) ---
      // This is the key security change.

      toast({
        title: 'Login Successful',
        description: 'Welcome back!',
      });
      navigate('/dashboard');
    } catch (err: any) {
      const errorMessage = err.message || 'An unknown error occurred';
      // --- UPDATED: Set error state for local display ---
      setError(errorMessage); 
      
      // Keep toast for general notification
      toast({
        title: 'Login Failed',
        description: errorMessage,
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <Card className="w-full max-w-sm shadow-xl">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold">Digital Twin Login</CardTitle>
          <CardDescription>Enter your credentials to access the dashboard</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* --- NEW: Improved Error Display --- */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Login Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                type="text"
                placeholder="admin"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={loading || demoMode} // Disable if demo is checked
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading || demoMode} // Disable if demo is checked
                required
              />
            </div>
            
            {/* --- NEW: Demo Mode Checkbox --- */}
            <div className="flex items-center space-x-2 pt-2">
              <Checkbox 
                id="demo-mode" 
                checked={demoMode}
                onCheckedChange={(checked) => setDemoMode(checked as boolean)}
                disabled={loading}
              />
              <Label
                htmlFor="demo-mode"
                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Launch Demo Mode (Offline)
              </Label>
            </div>
            
            <Button type="submit" className="w-full" disabled={loading}>
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {demoMode ? (
                <>
                  <Wand2 className="mr-2 h-4 w-4" />
                  Start Demo
                </>
              ) : (
                'Login'
              )}
            </Button>
          </form>
        </CardContent>
        <CardFooter>
            <p className="text-xs text-muted-foreground text-center w-full">
              {demoMode 
                ? 'Demo mode runs locally using simulated data.' 
                : 'Login securely (HttpOnly session)'}
            </p>
        </CardFooter>
      </Card>
    </div>
  );
}