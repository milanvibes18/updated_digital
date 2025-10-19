// src/components/AlertFeed.tsx
import { useState, useEffect, useCallback } from 'react';
import { AlertCard } from './AlertCard';
import { Alert as AlertType } from '../types/digital-twin';
import { getAlerts, acknowledgeAlert } from '../utils/api'; // Import API functions
import { useToast } from '../hooks/use-toast';
import { Loader2, ServerCrash, BellOff } from 'lucide-react';
import io, { Socket } from 'socket.io-client'; // Import socket.io client

// --- TODO: REPLACE WITH YOUR BACKEND DETAILS ---
const API_ALERTS_ENDPOINT = '/alerts'; // Endpoint for GET (history) and POST (acknowledge)
const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || 'http://127.0.0.1:5000'; // Your WebSocket server URL
const WEBSOCKET_PATH = '/socket.io'; // Standard Socket.IO path, adjust if needed
// ------------------------------------------

interface AlertFeedProps {
  userId: string; // Assuming alerts are user-specific
}

export function AlertFeed({ userId }: AlertFeedProps) {
  const [alerts, setAlerts] = useState<AlertType[]>([]);
  const [status, setStatus] = useState<'loading' | 'error' | 'success'>('loading');
  const [socket, setSocket] = useState<Socket | null>(null);
  const { toast } = useToast();

  const loadAlertHistory = useCallback(async () => {
    try {
      setStatus('loading');
      // Fetch initial history (e.g., last 50 alerts)
      const historicalAlerts = await getAlerts(userId, 50); // Use API function
      setAlerts(historicalAlerts);
      setStatus('success');
    } catch (error) {
      console.error('Error loading alert history:', error);
      setStatus('error');
      toast({
        title: 'Error Loading Alerts',
        description: 'Could not fetch alert history. Please try again later.',
        variant: 'destructive',
      });
    }
  }, [userId, toast]);

  // Effect 1: Fetch initial alert history on component mount
  useEffect(() => {
    loadAlertHistory();
  }, [loadAlertHistory]);

  // Effect 2: Connect to WebSocket for real-time updates
  useEffect(() => {
    // Retrieve token for WebSocket authentication
    const token = localStorage.getItem('jwt_token');
    if (!token) {
      console.error('No JWT token found for WebSocket connection.');
      // Optionally handle redirect to login or show error
      return;
    }

    // Initialize Socket.IO connection
    const newSocket = io(WEBSOCKET_URL, {
      path: WEBSOCKET_PATH,
      transports: ['websocket'], // Prefer WebSocket
      auth: { token }, // Send token for authentication
      // Add other options like reconnection attempts if needed
      reconnectionAttempts: 5,
      reconnectionDelay: 3000,
    });

    newSocket.on('connect', () => {
      console.log('WebSocket connected:', newSocket.id);
      setSocket(newSocket);
      // Optional: Subscribe to specific alert events if your backend uses rooms
      newSocket.emit('subscribe', { type: 'alerts', userId });
    });

    newSocket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      setSocket(null);
      // Optionally show a toast or indicator about disconnection
      if (reason !== 'io client disconnect') { // Don't show on manual disconnect/logout
        toast({
          title: 'Real-time Disconnected',
          description: 'Attempting to reconnect...',
          variant: 'destructive',
        });
      }
    });

    newSocket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      toast({
        title: 'Real-time Connection Error',
        description: `Could not connect to the real-time server: ${error.message}`,
        variant: 'destructive',
      });
    });

    // Listen for new alerts pushed from the server
    newSocket.on('alert_update', (newAlert: AlertType) => {
      console.log('Received new alert via WebSocket:', newAlert);
      setAlerts((prevAlerts) => {
        // Avoid adding duplicates if already received via history fetch
        if (prevAlerts.some((a) => a.id === newAlert.id)) {
          return prevAlerts;
        }
        // Add the new alert to the top, maintaining a reasonable limit (e.g., 100)
        return [newAlert, ...prevAlerts].slice(0, 100);
      });
      // Optionally trigger a notification/toast for new critical alerts
      if (newAlert.severity === 'critical' && !newAlert.acknowledged) {
        toast({
          title: `🚨 Critical Alert: ${newAlert.title}`,
          description: newAlert.message,
          variant: 'destructive',
          duration: 10000, // Show for longer
        });
      }
    });

    // Listen for acknowledgment updates pushed from the server (optional)
    // This handles cases where another user acknowledges the alert
    newSocket.on('alert_acknowledged', (data: { alertId: string; acknowledgedBy: string }) => {
       console.log('Received acknowledgment update via WebSocket:', data);
       setAlerts((prevAlerts) =>
         prevAlerts.map((alert) =>
           alert.id === data.alertId ? { ...alert, acknowledged: true } : alert
         )
       );
    });

    // Clean up the connection when the component unmounts or userId changes
    return () => {
      console.log('Cleaning up WebSocket connection.');
      newSocket.off('connect');
      newSocket.off('disconnect');
      newSocket.off('connect_error');
      newSocket.off('alert_update');
      newSocket.off('alert_acknowledged');
      newSocket.disconnect();
      setSocket(null);
    };
  }, [userId, toast]); // Reconnect if userId changes

  // Handler: Acknowledge an alert via API
  const handleAcknowledge = useCallback(async (alertId: string) => {
    try {
      // Step 1: Tell the backend via API
      await acknowledgeAlert(alertId); // Use API function

      // Step 2: Update the UI state immediately (optimistic update)
      setAlerts((prevAlerts) =>
        prevAlerts.map((alert) =>
          alert.id === alertId ? { ...alert, acknowledged: true } : alert
        )
      );

      toast({
        title: 'Alert Acknowledged',
        description: `Alert ${alertId} has been marked as acknowledged.`,
      });

      // Optional: Emit acknowledgment back via WebSocket if needed,
      // although the backend might broadcast this already
      // if (socket) {
      //   socket.emit('acknowledge_alert', { alertId, userId });
      // }

    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
      toast({
        title: 'Error Acknowledging Alert',
        description: 'Could not update alert status. Please try again.',
        variant: 'destructive',
      });
      // Optional: Revert optimistic update if API call fails
      // loadAlertHistory(); // Or revert just the specific alert
    }
  }, [toast, userId]); // Include socket if using emit

  // --- Render Logic ---
  if (status === 'loading') {
    return (
      <div className="flex items-center justify-center p-8 text-muted-foreground">
        <Loader2 className="h-6 w-6 animate-spin mr-2" />
        <span>Loading Alerts...</span>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-destructive">
        <ServerCrash className="h-10 w-10 mb-3" />
        <span className="font-medium">Failed to Load Alert Feed</span>
        <p className="text-sm text-muted-foreground mt-1">
          Could not connect to the server. Please check your connection or try again later.
        </p>
        <Button onClick={loadAlertHistory} variant="outline" size="sm" className="mt-4">
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2 pb-4">
      {alerts.length === 0 ? (
        <div className="text-center text-muted-foreground p-8 rounded-lg border border-dashed">
          <BellOff className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p className="font-medium">No Alerts</p>
          <p className="text-sm">There are currently no alerts to display.</p>
        </div>
      ) : (
        alerts.map((alert) => (
          <AlertCard
            key={alert.id} // The 'key' prop is essential for React lists
            alert={alert}
            onAcknowledge={handleAcknowledge}
          />
        ))
      )}
    </div>
  );
}