// src/components/Dashboard.tsx
import { useState, useEffect, useCallback } from 'react'; // Added useCallback
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { KPICard } from './KPICard';
import { DeviceCard } from './DeviceCard';
// --- REMOVED AlertCard import (now handled by AlertFeed) ---
import { ThresholdSettings } from './ThresholdSettings';
import { EmailLogsView } from './EmailLogsView';
// --- NEW: Import new views and AlertFeed ---
import { HistoricalAnalyticsView } from './HistoricalAnalyticsView'; // Assume this exists
import { AssetManagementView } from './AssetManagementView'; // Assume this exists
import { AlertFeed } from './AlertFeed'; // Import the new feed component
import { AlertsHistoryView } from './AlertsHistoryView'; // Assume this exists

import {
  Activity, Cpu, Zap, TrendingUp, Wifi, RefreshCw,
  Download, FileText, Bell, Settings, Mail,
  History, Database, BellRing, // Icons for tabs
  ServerCrash // Icon for error state
} from 'lucide-react';
import { Device, Alert, DashboardData, TimeRange } from '../types/digital-twin';
import {
  getDashboardData,
  // --- REMOVED acknowledgeAlert (handled by AlertFeed) ---
  generateReport,
  exportData,
  getCurrentUser // Import function to get user identity
} from '../utils/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'; // Added Legend
import { cn } from '../utils/cn';
import { useToast } from '../hooks/use-toast';
import { Button } from './ui/button';
import { useNavigate } from 'react-router-dom'; // Import for redirecting on auth error

export function Dashboard() {
  // --- State for data received from the API/WebSocket ---
  // --- REMOVED devices and alerts state (now managed within specific components like AlertFeed) ---
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);

  // --- Existing state ---
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');
  const [loading, setLoading] = useState(true);
  const [initialLoadError, setInitialLoadError] = useState<string | null>(null); // For initial load errors
  const [userId, setUserId] = useState<string | null>(null); // Store fetched userId
  const [lastUpdate, setLastUpdate] = useState(Date.now());
  const [showThresholds, setShowThresholds] = useState(false);
  const [showEmailLogs, setShowEmailLogs] = useState(false);
  // --- REMOVED newCriticalAlerts (handled within AlertFeed/AlertsHistoryView) ---
  const [isProcessing, setIsProcessing] = useState(false); // Combined state for export/report

  // --- NEW: State for active tab ---
  const [activeView, setActiveView] = useState('realtime');

  const { toast } = useToast();
  const navigate = useNavigate(); // Hook for navigation

  const timeRangeOptions: { label: string; value: TimeRange }[] = [
    { label: '1H', value: '1h' },
    { label: '4H', value: '4h' },
    { label: '24H', value: '24h' },
    { label: '7D', value: '7d' },
    { label: '30D', value: '30d' },
  ];

  // --- Effect to fetch User ID ---
  useEffect(() => {
    const fetchUser = async () => {
      try {
        const user = await getCurrentUser();
        if (user && user.logged_in_as) {
          setUserId(user.logged_in_as);
        } else {
          throw new Error('Could not verify user identity.');
        }
      } catch (error) {
        console.error('Authentication error:', error);
        toast({
          title: 'Authentication Required',
          description: 'Please log in to access the dashboard.',
          variant: 'destructive',
        });
        navigate('/login'); // Redirect to login if user fetch fails
      }
    };
    fetchUser();
  }, [navigate, toast]);


  // --- WebSocket and Initial Load Effect ---
  // Simplified: Dashboard primarily focuses on initial load and KPI/Chart updates.
  // AlertFeed handles its own WebSocket logic.
  useEffect(() => {
    if (!userId) return; // Don't load data until userId is available

    // --- Callback for loading data ---
    const loadInitialData = async () => {
      try {
        setLoading(true);
        setInitialLoadError(null); // Clear previous errors
        const data = await getDashboardData(); // No params needed if API knows user from token
        setDashboardData(data);
        setLastUpdate(Date.now());
      } catch (error: any) {
        console.error('Error loading initial dashboard data:', error);
        setInitialLoadError(error.message || 'Failed to load dashboard data.');
        // Don't show toast here, rely on the error state rendering
      } finally {
        setLoading(false);
      }
    };

    loadInitialData(); // Initial data load

    // --- Event Listener for general data updates (if needed beyond AlertFeed) ---
    // Example: If WebSocket pushes general dashboard KPIs
    const handleSocketUpdate = (event: Event) => {
      const customEvent = event as CustomEvent<DashboardData>;
      console.log('Dashboard received socket data update:', customEvent.detail);
      setDashboardData(prevData => ({ ...prevData, ...customEvent.detail }));
      setLastUpdate(Date.now());
    };

    document.addEventListener('socketDataUpdate', handleSocketUpdate as EventListener);

    // Optional: Set up a fallback polling mechanism if WebSocket fails or isn't used for KPIs
    const fallbackInterval = setInterval(() => {
       // Only poll if not recently updated by WebSocket (e.g., > 10 seconds ago)
      if (Date.now() - lastUpdate > 10000 && !loading && userId) {
         console.log('Polling for dashboard data...');
         loadInitialData(); // Re-fetch data
      }
    }, 15000); // Poll every 15 seconds

    return () => {
      document.removeEventListener('socketDataUpdate', handleSocketUpdate as EventListener);
      clearInterval(fallbackInterval);
    };
    // Re-run if userId changes (e.g., re-login) or timeRange changes
  }, [userId, timeRange, lastUpdate, loading]); // Added dependencies

  // --- Handlers ---

  const handleRefresh = useCallback(() => {
    if (!userId) return;
    setLoading(true);
    setInitialLoadError(null);
    getDashboardData() // Re-fetch data
      .then(data => {
        setDashboardData(data);
        setLastUpdate(Date.now());
        toast({
          title: "Data Refreshed",
          description: "Dashboard data has been updated.",
        });
      })
      .catch(error => {
        console.error('Error on manual refresh:', error);
        setInitialLoadError(error.message || 'Failed to refresh dashboard data.');
        toast({
          title: "Refresh Failed",
          description: "Could not retrieve latest data.",
          variant: "destructive",
        });
      })
      .finally(() => setLoading(false));
  }, [userId, toast]);

  const handleExport = useCallback(async (format: 'csv' | 'json') => {
    if (!userId) return;
    setIsProcessing(true);
    try {
      const result = await exportData(format, 7); // Export last 7 days for example
      // Trigger download
      const a = document.createElement('a');
      // --- IMPORTANT: Construct full URL if paths are relative ---
      // Assuming API_BASE_URL is like 'http://host/api'
      const downloadUrl = `${API_BASE_URL.replace('/api', '')}${result.export_path}`;
      a.href = downloadUrl;
      a.download = result.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      toast({ title: 'Export Successful', description: `Data exported to ${format.toUpperCase()}.` });
    } catch (error: any) {
      console.error('Error exporting data:', error);
      toast({
        title: 'Export Failed',
        description: error.message || 'An error occurred during export.',
        variant: 'destructive',
      });
    } finally {
      setIsProcessing(false);
    }
  }, [userId, toast]);

  const handleGenerateReport = useCallback(async () => {
    if (!userId) return;
    setIsProcessing(true);
    try {
      const result = await generateReport();
      // Open in new tab
      // --- IMPORTANT: Construct full URL if paths are relative ---
      const reportUrl = `${API_BASE_URL.replace('/api', '')}${result.report_path}`;
      window.open(reportUrl, '_blank');

      toast({ title: 'Report Generated', description: 'Report opened in a new tab.' });
    } catch (error: any) {
      console.error('Error generating report:', error);
      toast({
        title: 'Report Generation Failed',
        description: error.message || 'An error occurred.',
        variant: 'destructive',
      });
    } finally {
      setIsProcessing(false);
    }
  }, [userId, toast]);


  // --- Render Loading State for Initial Load ---
  if (!userId || (loading && !dashboardData && !initialLoadError)) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex items-center space-x-2 text-muted-foreground">
          <Loader2 className="h-6 w-6 animate-spin" />
          <span>Loading Dashboard...</span>
        </div>
      </div>
    );
  }

  // --- Render Error State for Initial Load ---
   if (initialLoadError) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-6">
         <Card className="w-full max-w-md border-destructive">
           <CardHeader>
             <CardTitle className="flex items-center gap-2 text-destructive">
               <ServerCrash className="h-6 w-6" />
               Dashboard Load Failed
             </CardTitle>
             <CardDescription>
               We couldn't load the initial dashboard data.
             </CardDescription>
           </CardHeader>
           <CardContent>
             <p className="text-sm text-muted-foreground mb-4">
               Error: {initialLoadError}
             </p>
             <p className="text-sm text-muted-foreground">
               Please check your network connection or try refreshing. If the problem persists, contact support.
             </p>
           </CardContent>
           <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={() => navigate('/login')}>Go to Login</Button>
              <Button onClick={handleRefresh} disabled={loading}>
                <RefreshCw className={cn("h-4 w-4 mr-2", loading && "animate-spin")} />
                Retry
              </Button>
           </CardFooter>
         </Card>
      </div>
    );
  }

  // --- Render Dashboard Content ---
  // Use displayData fallback for safety, though errors/loading should prevent reaching here without data
   const displayData = dashboardData || { /* Default structure */ } as DashboardData;

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-secondary/10 to-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/70 backdrop-blur-lg sticky top-0 z-40 shadow-sm">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4"> {/* Responsive padding */}
          <div className="flex flex-wrap items-center justify-between gap-4"> {/* Flex wrap for smaller screens */}
            <div className="flex items-center gap-3">
              <Activity className="h-7 w-7 text-primary" />
              <h1 className="text-2xl font-bold tracking-tight">
                Digital Twin
              </h1>
            </div>

            <div className="flex items-center gap-2 sm:gap-4 flex-wrap justify-end"> {/* Flex wrap for controls */}
              {/* Connection Status */}
              <div className="flex items-center gap-1.5 text-xs sm:text-sm text-muted-foreground">
                <Wifi className="h-4 w-4 text-green-500" />
                <span>Connected</span>
                <span className="hidden sm:inline text-muted-foreground/70 ml-1">
                  | Last: {new Date(lastUpdate).toLocaleTimeString()}
                </span>
              </div>

              {/* Action Buttons */}
               <div className="flex items-center gap-1 sm:gap-2">
                 <Button variant="ghost" size="sm" onClick={() => setShowThresholds(true)} aria-label="Settings">
                   <Settings className="h-4 w-4" />
                   <span className="hidden lg:inline ml-1">Settings</span>
                 </Button>
                 <Button variant="ghost" size="sm" onClick={() => setShowEmailLogs(true)} aria-label="Email Logs">
                   <Mail className="h-4 w-4" />
                   <span className="hidden lg:inline ml-1">Logs</span>
                 </Button>
                 <Button variant="outline" size="sm" onClick={handleRefresh} disabled={loading} aria-label="Refresh Data">
                   <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
                   <span className="hidden lg:inline ml-1">Refresh</span>
                 </Button>
                 <Button variant="outline" size="sm" onClick={() => handleExport('csv')} disabled={isProcessing} aria-label="Export CSV">
                   <Download className="h-4 w-4" />
                   <span className="hidden lg:inline ml-1">CSV</span>
                 </Button>
                 <Button size="sm" onClick={handleGenerateReport} disabled={isProcessing} aria-label="Generate Report">
                   <FileText className="h-4 w-4" />
                   <span className="hidden lg:inline ml-1">Report</span>
                 </Button>
               </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
         {/* Time Range Selector */}
         <div className="mb-6 sm:mb-8">
           <div className="flex flex-wrap items-center gap-1 sm:gap-2">
             {timeRangeOptions.map(option => (
               <Button
                 key={option.value}
                 variant={timeRange === option.value ? 'default' : 'secondary'}
                 size="sm"
                 onClick={() => setTimeRange(option.value)}
                 className="text-xs sm:text-sm"
               >
                 {option.label}
               </Button>
             ))}
           </div>
         </div>

        {/* Tabbed Interface */}
        <Tabs value={activeView} onValueChange={setActiveView} className="w-full">
           <TabsList className="grid w-full grid-cols-2 sm:grid-cols-4 mb-6 sm:mb-8 text-xs sm:text-sm">
             <TabsTrigger value="realtime">
               <Activity className="h-4 w-4 mr-1 sm:mr-2" />
               Real-time
             </TabsTrigger>
             <TabsTrigger value="historical">
               <History className="h-4 w-4 mr-1 sm:mr-2" />
               Analytics
             </TabsTrigger>
             <TabsTrigger value="assets">
               <Database className="h-4 w-4 mr-1 sm:mr-2" />
               Assets
             </TabsTrigger>
             <TabsTrigger value="alerts">
               <BellRing className="h-4 w-4 mr-1 sm:mr-2" />
               Alerts
             </TabsTrigger>
           </TabsList>

          {/* Real-time Tab */}
          <TabsContent value="realtime">
            {/* KPI Cards */}
             <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-6 sm:mb-8">
              <KPICard title="System Health" value={`${displayData.systemHealth?.toFixed(1) || 'N/A'}%`} icon={Activity} variant={displayData.systemHealth >= 80 ? 'success' : displayData.systemHealth >= 60 ? 'warning' : 'danger'} loading={loading} />
              <KPICard title="Active Devices" value={`${displayData.activeDevices || 0} / ${displayData.totalDevices || 0}`} subtitle={`${displayData.statusDistribution?.offline || 0} offline`} icon={Cpu} loading={loading} />
              <KPICard title="Energy Usage" value={`${displayData.energyUsage?.toFixed(1) || 'N/A'} kW`} subtitle={`$${displayData.energyCost?.toFixed(2) || 'N/A'} / hr`} icon={Zap} variant="warning" loading={loading} />
              <KPICard title="Avg Efficiency" value={`${displayData.efficiency?.toFixed(1) || 'N/A'}%`} icon={TrendingUp} variant="info" loading={loading} />
            </div>

            {/* Main Chart and Alert Feed Grid */}
             <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8">
               {/* Performance Chart */}
               <div className="lg:col-span-2">
                 <Card className="shadow-md hover:shadow-lg transition-shadow">
                   <CardHeader>
                     <CardTitle className="text-lg sm:text-xl">Performance Metrics</CardTitle>
                     <CardDescription>System health and efficiency over time ({timeRange})</CardDescription>
                   </CardHeader>
                   <CardContent>
                     <div className="h-[300px] sm:h-[400px]"> {/* Responsive height */}
                       <ResponsiveContainer width="100%" height="100%">
                         <LineChart data={displayData.performanceData || []} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}> {/* Adjusted margins */}
                           <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
                           <XAxis dataKey="timestamp" stroke="hsl(var(--muted-foreground))" fontSize={10} tickFormatter={(ts) => new Date(ts).toLocaleTimeString()} />
                           <YAxis stroke="hsl(var(--muted-foreground))" fontSize={10} domain={[0, 100]} unit="%" />
                           <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--popover))', border: '1px solid hsl(var(--border))', borderRadius: 'var(--radius)' }} labelStyle={{ color: 'hsl(var(--foreground))' }} itemStyle={{ fontSize: '12px' }} />
                           <Legend verticalAlign="top" height={36} iconSize={10} wrapperStyle={{ fontSize: '12px' }}/>
                           <Line type="monotone" dataKey="systemHealth" stroke="hsl(var(--success))" strokeWidth={2} dot={false} name="Health" unit="%" />
                           <Line type="monotone" dataKey="efficiency" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} name="Efficiency" unit="%" />
                         </LineChart>
                       </ResponsiveContainer>
                     </div>
                   </CardContent>
                 </Card>
               </div>

              {/* Alert Feed */}
              <div className="lg:col-span-1">
                 <Card className="shadow-md hover:shadow-lg transition-shadow h-full flex flex-col"> {/* Ensure card takes height */}
                   <CardHeader>
                     <CardTitle className="flex items-center gap-2 text-lg sm:text-xl">
                       <Bell className="h-5 w-5" /> Recent Alerts
                     </CardTitle>
                     <CardDescription>Live feed of system alerts</CardDescription>
                   </CardHeader>
                   <CardContent className="flex-grow overflow-hidden"> {/* Allow content to grow and hide overflow */}
                     {/* --- Integrate AlertFeed --- */}
                     {userId ? (
                       <AlertFeed userId={userId} />
                     ) : (
                       <div className="flex items-center justify-center h-full text-muted-foreground">
                         <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading user...
                       </div>
                     )}
                   </CardContent>
                 </Card>
               </div>
            </div>

            {/* Devices Grid */}
             <div className="mt-6 sm:mt-8">
               <Card className="shadow-md hover:shadow-lg transition-shadow">
                 <CardHeader>
                   <CardTitle className="text-lg sm:text-xl">Device Status Overview</CardTitle>
                   <CardDescription>Real-time monitoring of connected devices</CardDescription>
                 </CardHeader>
                 <CardContent>
                   <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4 sm:gap-6">
                     {(displayData.devices || []).map(device => (
                       <DeviceCard
                         key={device.id}
                         device={device}
                         // Optional: Add onClick navigation or modal
                         // onClick={() => navigate(`/device/${device.id}`)}
                       />
                     ))}
                     {(displayData.devices || []).length === 0 && !loading && (
                        <p className="col-span-full text-center text-muted-foreground py-8">No devices to display.</p>
                     )}
                   </div>
                 </CardContent>
               </Card>
             </div>
          </TabsContent>

          {/* Historical Analytics Tab */}
          <TabsContent value="historical">
             {userId ? (
               <HistoricalAnalyticsView userId={userId} timeRange={timeRange} />
             ) : (
                <div className="text-center p-8 text-muted-foreground">Loading user data...</div>
             )}
          </TabsContent>

          {/* Asset Management Tab */}
          <TabsContent value="assets">
             {userId ? (
                <AssetManagementView userId={userId} />
             ) : (
                 <div className="text-center p-8 text-muted-foreground">Loading user data...</div>
             )}
          </TabsContent>

          {/* All Alerts Tab */}
          <TabsContent value="alerts">
             {userId ? (
                // Assuming AlertsHistoryView fetches its own data or uses AlertFeed's data via context/props
                <AlertsHistoryView userId={userId} />
             ) : (
                 <div className="text-center p-8 text-muted-foreground">Loading user data...</div>
             )}
          </TabsContent>
        </Tabs>
      </main>

      {/* Settings Modal */}
      {showThresholds && userId && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <ThresholdSettings
            userId={userId}
            onClose={() => setShowThresholds(false)}
          />
        </div>
      )}

      {/* Email Logs Modal */}
      {showEmailLogs && userId && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <EmailLogsView
            userId={userId}
            onClose={() => setShowEmailLogs(false)}
          />
        </div>
      )}
    </div>
  );
}

// --- Placeholder Components for missing views ---
// Replace these with your actual component implementations

// function HistoricalAnalyticsView({ userId, timeRange }: { userId: string, timeRange: TimeRange }) {
//   return <Card><CardHeader><CardTitle>Historical Analytics</CardTitle></CardHeader><CardContent><p>Historical Analytics View for user {userId} ({timeRange}) - Component not implemented yet.</p></CardContent></Card>;
// }

// function AssetManagementView({ userId }: { userId: string }) {
//   return <Card><CardHeader><CardTitle>Asset Management</CardTitle></CardHeader><CardContent><p>Asset Management View for user {userId} - Component not implemented yet.</p></CardContent></Card>;
// }

// function AlertsHistoryView({ userId }: { userId: string }) {
//   const [alerts, setAlerts] = useState<Alert[]>([]);
//   useEffect(() => {
//     getAlerts(userId, 100).then(setAlerts); // Fetch history
//   }, [userId]);
//   return <Card><CardHeader><CardTitle>All Alerts History</CardTitle></CardHeader><CardContent className="space-y-4 max-h-[70vh] overflow-y-auto"><p>Alerts History View for user {userId} - Component not fully implemented yet.</p>{alerts.map(a => <AlertCard key={a.id} alert={a} />)}</CardContent></Card>;
// }