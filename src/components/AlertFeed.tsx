// src/components/AlertFeed.tsx
import { useState, useEffect, useCallback, useMemo } from 'react'; // Added useMemo
import { AlertCard } from './AlertCard';
import { Alert as AlertType } from '../types/digital-twin';
// Removed unused getAlerts, acknowledgeAlert from api (handled by useDashboard or WebSocket)
import { useToast } from '../hooks/use-toast';
import { Loader2, ServerCrash, BellOff, ArrowDownUp, Filter } from 'lucide-react'; // Added icons
import io, { Socket } from 'socket.io-client';
import { Button } from './ui/button'; // Added Button
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'; // Added Select
import { Label } from './ui/label'; // Added Label

// WebSocket configuration (Ensure VITE_WEBSOCKET_URL is set in your .env)
const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || 'http://127.0.0.1:5000';
const WEBSOCKET_PATH = '/socket.io';

interface AlertFeedProps {
  userId: string;
  initialAlerts?: AlertType[]; // Allow passing initial alerts
  onAcknowledge?: (alertId: string) => void; // Prop for handling acknowledgement
}

type SortOrder = 'newest' | 'oldest' | 'severity';
type FilterSeverity = 'all' | 'critical' | 'warning' | 'info';

export function AlertFeed({ userId, initialAlerts = [], onAcknowledge }: AlertFeedProps) {
  const [alerts, setAlerts] = useState<AlertType[]>(initialAlerts);
  const [status, setStatus] = useState<'loading' | 'error' | 'success'>(initialAlerts.length > 0 ? 'success' : 'loading');
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false); // Track connection status
  const { toast } = useToast();

  // --- NEW: Sorting and Filtering State ---
  const [sortOrder, setSortOrder] = useState<SortOrder>('newest');
  const [filterSeverity, setFilterSeverity] = useState<FilterSeverity>('all');
  // -----------------------------------------

  // --- Load initial history (can be simplified if useDashboard provides it) ---
  const loadAlertHistory = useCallback(async () => {
    // This function might be redundant if useDashboard hook provides initial alerts.
    // Keeping it for standalone potential or direct API call if needed.
    // Simulating initial load if `initialAlerts` is empty
    if (alerts.length === 0) {
      setStatus('loading');
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 500));
      // In a real scenario, fetch from API here if needed
      // const historicalAlerts = await getAlerts(userId, 50);
      // setAlerts(historicalAlerts);
      setStatus('success'); // Assume success for demo if no initial alerts provided
    } else {
      setStatus('success');
    }
  }, [userId, toast, alerts.length]);

  // Effect 1: Fetch initial alert history on component mount (if needed)
  useEffect(() => {
    if (initialAlerts.length === 0) {
      loadAlertHistory();
    }
  }, [loadAlertHistory, initialAlerts.length]);


  // Effect 2: Connect to WebSocket for real-time updates
  useEffect(() => {
    const token = localStorage.getItem('jwt_token'); // Or however you manage auth

    // --- Conditional Connection ---
    // Prevent connection in demo mode or if already connected
    if (localStorage.getItem('demo_mode') === 'true' || socket || !token) {
        if (!token) console.error("AlertFeed: No auth token found for WebSocket.");
        return;
    }
    // ----------------------------

    console.log(`AlertFeed: Attempting WebSocket connection to ${WEBSOCKET_URL}`);
    const newSocket = io(WEBSOCKET_URL, {
      path: WEBSOCKET_PATH,
      transports: ['websocket'],
      auth: { token },
      reconnectionAttempts: 5,
      reconnectionDelay: 3000,
    });

    newSocket.on('connect', () => {
      console.log('AlertFeed WebSocket connected:', newSocket.id);
      setSocket(newSocket);
      setIsConnected(true);
      newSocket.emit('subscribe', { type: 'alerts', userId }); // Example subscription
    });

    newSocket.on('disconnect', (reason) => {
      console.log('AlertFeed WebSocket disconnected:', reason);
      setSocket(null);
      setIsConnected(false);
      if (reason !== 'io client disconnect') {
        // toast({ title: 'Real-time Disconnected', description: 'Attempting alert feed reconnect...', variant: 'destructive' });
      }
    });

    newSocket.on('connect_error', (error) => {
      console.error('AlertFeed WebSocket connection error:', error);
      setIsConnected(false);
      setStatus('error'); // Set error status if connection fails initially
      // toast({ title: 'Alert Feed Connection Error', description: `Could not connect: ${error.message}`, variant: 'destructive' });
    });

    // Listen for new alerts
    newSocket.on('alert_update', (newAlert: AlertType) => {
      console.log('AlertFeed received alert_update:', newAlert);
      setAlerts((prevAlerts) => {
        if (prevAlerts.some((a) => a.id === newAlert.id)) {
          // If alert exists, update it (e.g., if status changed)
          return prevAlerts.map(a => a.id === newAlert.id ? newAlert : a);
        }
        // Add new alert to the top
        return [newAlert, ...prevAlerts].slice(0, 100); // Limit history size
      });
      // Notification handled by useDashboard or parent component
    });

    // Listen for acknowledgment updates (if backend broadcasts)
    newSocket.on('alert_acknowledged', (data: { alertId: string; acknowledgedBy?: string }) => {
       console.log('AlertFeed received acknowledgment:', data);
       setAlerts((prevAlerts) =>
         prevAlerts.map((alert) =>
           alert.id === data.alertId ? { ...alert, acknowledged: true } : alert
         )
       );
    });

    // Cleanup
    return () => {
      console.log('Cleaning up AlertFeed WebSocket.');
      newSocket.off('connect');
      newSocket.off('disconnect');
      newSocket.off('connect_error');
      newSocket.off('alert_update');
      newSocket.off('alert_acknowledged');
      newSocket.disconnect();
      setSocket(null);
      setIsConnected(false);
    };
  }, [userId, toast, socket]); // Rerun if userId changes or socket needs re-init


  // --- NEW: Sorting and Filtering Logic ---
  const sortedAndFilteredAlerts = useMemo(() => {
    let processedAlerts = [...alerts];

    // Filter
    if (filterSeverity !== 'all') {
      processedAlerts = processedAlerts.filter(alert => alert.severity === filterSeverity);
    }

    // Sort
    const severityOrder: Record<AlertType['severity'], number> = { critical: 0, warning: 1, info: 2 };
    processedAlerts.sort((a, b) => {
      switch (sortOrder) {
        case 'oldest':
          return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
        case 'severity':
           // Sort by severity first (critical -> warning -> info), then by time (newest first)
          const severityDiff = severityOrder[a.severity] - severityOrder[b.severity];
          if (severityDiff !== 0) return severityDiff;
          return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
        case 'newest':
        default:
          return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
      }
    });

    return processedAlerts;
  }, [alerts, sortOrder, filterSeverity]);
  // ---------------------------------------

  // --- Render Logic ---
  const renderContent = () => {
    if (status === 'loading') {
      return (
        <div className="flex items-center justify-center p-8 text-muted-foreground">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          <span>Loading Alerts...</span>
        </div>
      );
    }

    if (status === 'error' && !isConnected) { // Show error only if WS connection also failed
      return (
        <div className="flex flex-col items-center justify-center p-8 text-destructive">
          <ServerCrash className="h-10 w-10 mb-3" />
          <span className="font-medium">Failed to Load Alert Feed</span>
          <p className="text-sm text-muted-foreground mt-1">
            Could not connect to the real-time server.
          </p>
          {/* <Button onClick={loadAlertHistory} variant="outline" size="sm" className="mt-4">
            Retry History
          </Button> */}
        </div>
      );
    }

    if (sortedAndFilteredAlerts.length === 0) {
      return (
        <div className="text-center text-muted-foreground p-8 rounded-lg border border-dashed">
          <BellOff className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p className="font-medium">No Alerts</p>
          <p className="text-sm">
            {filterSeverity === 'all'
              ? 'There are currently no alerts to display.'
              : `No alerts match the severity "${filterSeverity}".`}
          </p>
        </div>
      );
    }

    return sortedAndFilteredAlerts.map((alert) => (
      <AlertCard
        key={alert.id}
        alert={alert}
        onAcknowledge={onAcknowledge} // Pass down the handler from props
      />
    ));
  };


  return (
    <div className="flex flex-col h-full">
        {/* --- NEW: Filter and Sort Controls --- */}
        <div className="flex items-center justify-between gap-4 p-4 border-b bg-card sticky top-0 z-10">
            <div className='flex items-center gap-2'>
                <Filter className="h-4 w-4 text-muted-foreground" />
                <Label htmlFor="filter-severity" className="text-sm font-medium">Severity:</Label>
                <Select
                    value={filterSeverity}
                    onValueChange={(value) => setFilterSeverity(value as FilterSeverity)}
                >
                    <SelectTrigger id="filter-severity" className="h-8 w-[120px] text-xs">
                        <SelectValue placeholder="Severity" />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="all">All</SelectItem>
                        <SelectItem value="critical">Critical</SelectItem>
                        <SelectItem value="warning">Warning</SelectItem>
                        <SelectItem value="info">Info</SelectItem>
                    </SelectContent>
                </Select>
            </div>
             <div className='flex items-center gap-2'>
                <ArrowDownUp className="h-4 w-4 text-muted-foreground" />
                <Label htmlFor="sort-order" className="text-sm font-medium">Sort By:</Label>
                 <Select
                    value={sortOrder}
                    onValueChange={(value) => setSortOrder(value as SortOrder)}
                >
                    <SelectTrigger id="sort-order" className="h-8 w-[120px] text-xs">
                        <SelectValue placeholder="Sort Order" />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="newest">Newest First</SelectItem>
                        <SelectItem value="oldest">Oldest First</SelectItem>
                        <SelectItem value="severity">Severity</SelectItem>
                    </SelectContent>
                </Select>
            </div>
             {/* Optional: Connection Status Indicator */}
             <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} title={isConnected ? 'Real-time connected' : 'Real-time disconnected'}></div>
        </div>
        {/* ------------------------------------- */}

      <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {renderContent()}
      </div>
    </div>
  );
}