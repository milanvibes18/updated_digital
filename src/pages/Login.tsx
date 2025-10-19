import { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { loginUser, registerUser } from '@/utils/api';
import { useToast } from '@/hooks/use-toast';
import { Activity, Loader2 } from 'lucide-react';

export function LoginPage() {
  // Login states
  const [loginUsername, setLoginUsername] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);

  // Registration states
  const [regUsername, setRegUsername] = useState('');
  const [regPassword, setRegPassword] = useState('');
  const [regEmail, setRegEmail] = useState('');
  const [registerLoading, setRegisterLoading] = useState(false);

  // Tabs control
  const [activeTab, setActiveTab] = useState<'login' | 'register'>('login');

  const navigate = useNavigate();
  const { toast } = useToast();

  // --- LOGIN HANDLER ---
  const handleLogin = async (e: FormEvent) => {
    e.preventDefault();
    setLoginLoading(true);

    try {
      await loginUser(loginUsername, loginPassword);
      toast({
        title: 'Login Successful',
        description: 'Redirecting to dashboard...',
      });
      navigate('/'); // Redirect after login
    } catch (error: unknown) {
      let description = 'Invalid credentials or server error.';
      if (error instanceof Error) {
        description = error.message;
      } else if (typeof error === 'object' && error && 'message' in (error as any)) {
        description = (error as any).message;
      }
      toast({
        title: 'Login Failed',
        description,
        variant: 'destructive',
      });
    } finally {
      setLoginLoading(false);
    }
  };

  // --- REGISTER HANDLER ---
  const handleRegister = async (e: FormEvent) => {
    e.preventDefault();
    setRegisterLoading(true);

    try {
      await registerUser(regUsername, regPassword, regEmail);
      toast({
        title: 'Registration Successful',
        description: 'Please log in with your new credentials.',
      });
      // Switch back to login tab
      setActiveTab('login');
      // Clear registration fields
      setRegUsername('');
      setRegPassword('');
      setRegEmail('');
    } catch (error: unknown) {
      let description = 'Username or email may already exist, or server error.';
      if (error instanceof Error) {
        description = error.message;
      } else if (typeof error === 'object' && error && 'message' in (error as any)) {
        description = (error as any).message;
      }
      toast({
        title: 'Registration Failed',
        description,
        variant: 'destructive',
      });
    } finally {
      setRegisterLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <Tabs value={activeTab} onValueChange={(val) => setActiveTab(val as 'login' | 'register')} className="w-[400px]">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="login">Login</TabsTrigger>
          <TabsTrigger value="register">Register</TabsTrigger>
        </TabsList>

        {/* --- LOGIN TAB --- */}
        <TabsContent value="login">
          <Card>
            <form onSubmit={handleLogin}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-6 w-6 text-primary" />
                  Digital Twin Login
                </CardTitle>
                <CardDescription>
                  Access your Industrial IoT Dashboard.
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="login-username">Username</Label>
                  <Input
                    id="login-username"
                    autoComplete="username"
                    value={loginUsername}
                    onChange={(e) => setLoginUsername(e.target.value)}
                    required
                    disabled={loginLoading}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="login-password">Password</Label>
                  <Input
                    id="login-password"
                    type="password"
                    autoComplete="current-password"
                    value={loginPassword}
                    onChange={(e) => setLoginPassword(e.target.value)}
                    required
                    disabled={loginLoading}
                  />
                </div>
              </CardContent>

              <CardFooter>
                <Button
                  type="submit"
                  className="w-full"
                  disabled={loginLoading}
                  aria-busy={loginLoading}
                >
                  {loginLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  {loginLoading ? 'Logging in...' : 'Login'}
                </Button>
              </CardFooter>
            </form>
          </Card>
        </TabsContent>

        {/* --- REGISTER TAB --- */}
        <TabsContent value="register">
          <Card>
            <form onSubmit={handleRegister}>
              <CardHeader>
                <CardTitle>Register</CardTitle>
                <CardDescription>
                  Create a new account to access the dashboard.
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="reg-username">Username</Label>
                  <Input
                    id="reg-username"
                    autoComplete="username"
                    value={regUsername}
                    onChange={(e) => setRegUsername(e.target.value)}
                    required
                    disabled={registerLoading}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="reg-email">
                    Email <span className="text-muted-foreground">(optional)</span>
                  </Label>
                  <Input
                    id="reg-email"
                    type="email"
                    autoComplete="email"
                    value={regEmail}
                    onChange={(e) => setRegEmail(e.target.value)}
                    disabled={registerLoading}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="reg-password">Password</Label>
                  <Input
                    id="reg-password"
                    type="password"
                    autoComplete="new-password"
                    value={regPassword}
                    onChange={(e) => setRegPassword(e.target.value)}
                    required
                    disabled={registerLoading}
                  />
                </div>
              </CardContent>

              <CardFooter>
                <Button
                  type="submit"
                  className="w-full"
                  disabled={registerLoading}
                  aria-busy={registerLoading}
                >
                  {registerLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  {registerLoading ? 'Registering...' : 'Register'}
                </Button>
              </CardFooter>
            </form>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
