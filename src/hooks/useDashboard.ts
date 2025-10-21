import { useState, useEffect, useRef, useCallback } from 'react';
import io, { Socket } from 'socket.io-client'; // Import socket.io client
import { blink } from '../blink/client'; // Assuming blink is still used for DB init/fallback
import { DataSimulator, defaultThresholds } from '../utils/data-simulator';
import {
  getDevices,
  getAlerts,
  generateSampleData,
  initializeDatabase,
  acknowledgeAlert,
} from '../utils/db'; // Keep DB utils if still used for init/fallback
import { Device, Alert, DashboardData, TimeRange } from '../types/digital-twin';
import { useToast } from '../hooks/use-toast';

// --- WebSocket Configuration (Adjust as needed) ---
const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || 'http://127.0.0.1:5000'; // Your WebSocket server URL
const WEBSOCKET_PATH = '/socket.io'; // Standard Socket.IO path
// --------------------------------------------------

export const useDashboard = () => {
  const [devices, setDevices] = useState<Device[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');
  const [loading, setLoading] = useState(true);
  const [userId] = useState('demo-user'); // Keep for initial data fetch or context
  const [lastUpdate, setLastUpdate] = useState(Date.now());
  const [showThresholds, setShowThresholds] = useState(false);
  const [showEmailLogs, setShowEmailLogs] = useState(false);
  const [newCriticalAlerts, setNewCriticalAlerts] = useState<Set<string>>(new Set());
  const [isExporting, setIsExporting] = useState(false);

  // --- WebSocket State ---
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  // -----------------------

  const simulatorRef = useRef<DataSimulator | null>(null);
  const { toast } = useToast();

  // --- Process incoming dashboard data (from WebSocket or initial load) ---
  const processDashboardUpdate = useCallback((devicesData: Device[], dashboardMetrics: Partial<DashboardData> | null) => {
    setDevices(devicesData);
    setLastUpdate(Date.now()); // Update timestamp on new data

    // --- Calculate derived metrics if not fully provided by WebSocket ---
    // If your WebSocket sends the full DashboardData structure, you might just use dashboardMetrics directly.
    // Otherwise, recalculate based on the received devices.
    const totalDevices = devicesData.length;
    const activeDevices = devicesData.filter((d) => d.status !== 'offline').length;
    const systemHealth =
      dashboardMetrics?.systemHealth ?? // Use from WS if available
      (totalDevices > 0
        ? (devicesData.reduce((avg, device) => avg + (device.healthScore || 0), 0) / totalDevices) * 100 // Handle potential missing score
        : 0);
    const efficiency =
      dashboardMetrics?.efficiency ?? // Use from WS if available
      (totalDevices > 0
        ? (devicesData.reduce((avg, device) => avg + (device.efficiencyScore || 0), 0) / totalDevices) * 100 // Handle potential missing score
        : 0);
    const energyUsage =
      dashboardMetrics?.energyUsage ?? // Use from WS if available
      devicesData
        .filter((d) => d.type === 'power_meter')
        .reduce((sum, device) => sum + (device.value || 0), 0); // Handle potential missing value

    // Generate performance data or use from WebSocket if provided
    const performanceData =
      dashboardMetrics?.performanceData && dashboardMetrics.performanceData.length > 0
        ? dashboardMetrics.performanceData
        : Array.from({ length: 24 }, (_, i) => ({ // Fallback generation
            timestamp: `${String(i).padStart(2, '0')}:00`,
            systemHealth: Math.max(0, Math.min(100, systemHealth + Math.sin(i * 0.2) * 10 + Math.random() * 5)),
            efficiency: Math.max(0, Math.min(100, efficiency + Math.cos(i * 0.15) * 8 + Math.random() * 4)),
            energyUsage: Math.max(0, energyUsage + Math.sin(i * 0.1) * energyUsage * 0.2),
          }));

    const statusDistribution =
      dashboardMetrics?.statusDistribution ?? // Use from WS if available
      { // Fallback calculation
        normal: devicesData.filter((d) => d.status === 'normal').length,
        warning: devicesData.filter((d) => d.status === 'warning').length,
        critical: devicesData.filter((d) => d.status === 'critical').length,
        offline: devicesData.filter((d) => d.status === 'offline').length,
      };

    setDashboardData({
      systemHealth: Math.round(systemHealth),
      activeDevices,
      totalDevices,
      efficiency: Math.round(efficiency),
      energyUsage: Math.round(energyUsage),
      energyCost: Math.round(energyUsage * 0.12), // $0.12 per kWh
      performanceData,
      statusDistribution,
      // --- Ensure these fields exist, even if empty ---
      devices: devicesData, // Add devices to dashboard data
      alerts: alerts, // Keep current alerts state
      timestamp: new Date().toISOString(), // Add timestamp
    });
    // --- End Metrics Calculation ---

  }, [alerts]); // Include alerts dependency

  // --- Load Initial Data (Only runs once or on manual refresh) ---
  const loadInitialData = useCallback(async () => {
    // setLoading(true); // Already handled in initializeApp
    try {
      // Fetch initial state from API (or DB if in demo/offline mode)
      const [initialDevices, initialAlerts] = await Promise.all([
        getDevices(userId),
        getAlerts(userId, 50), // Fetch more history initially
      ]);

      setAlerts(initialAlerts);
      // Process initial device data to populate dashboardData
      processDashboardUpdate(initialDevices, null); // Pass null as dashboardMetrics initially

    } catch (error) {
      console.error('Error loading initial data:', error);
      toast({
        title: 'Error',
        description: 'Failed to load initial dashboard data.',
        variant: 'destructive',
      });
    } finally {
      // setLoading(false); // Loading state managed by initializeApp
    }
  }, [userId, toast, processDashboardUpdate]);

  // --- Initialize App (DB, Demo Mode, Initial Data) ---
  const initializeApp = useCallback(async () => {
    try {
      setLoading(true);

      // --- Keep DB initialization if needed for demo/offline mode ---
      if (localStorage.getItem('demo_mode') === 'true') {
        await initializeDatabase();
        const existingDevices = await getDevices(userId);
        if (existingDevices.length === 0) {
          await generateSampleData(userId);
        }
        // Initialize thresholds if needed for demo
        const existingThresholds = await (blink.db as any).alertThresholds.list({ where: { userId } });
        if (existingThresholds.length === 0) {
          await Promise.all(
            defaultThresholds.map((threshold: any) =>
              (blink.db as any).alertThresholds.create({
                ...threshold,
                id: `THRESH_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                userId,
              })
            )
          );
        }
        // Start simulator only in demo mode
        simulatorRef.current = new DataSimulator(userId);
        simulatorRef.current.start();
      }
      // --- End Demo Mode Init ---

      await loadInitialData(); // Load initial data after setup

    } catch (error) {
      console.error('Error initializing app:', error);
      toast({
        title: 'Initialization Error',
        description: 'Failed to initialize the application.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false); // Set loading false after init and initial data load
    }
  }, [userId, loadInitialData, toast]);

  // --- Effect 1: Initialize App on Mount ---
  useEffect(() => {
    initializeApp();

    // Cleanup simulator on unmount if it was started
    return () => {
      if (simulatorRef.current) {
        simulatorRef.current.stop();
      }
    };
  }, [initializeApp]);

  // --- Effect 2: WebSocket Connection and Event Handling ---
  useEffect(() => {
    // Only connect if not in demo mode
    if (localStorage.getItem('demo_mode') === 'true') {
      console.log('Skipping WebSocket connection in Demo Mode.');
      return;
    }

    // Retrieve token for WebSocket authentication (check if needed based on backend)
    // Adjust this if your backend uses cookies (fetchWithCredentials) instead
    const token = localStorage.getItem('jwt_token');
    // if (!token) { // If token is absolutely required
    //   console.error('No JWT token found for WebSocket connection.');
    //   toast({ title: 'Auth Error', description: 'Cannot connect to real-time updates.', variant: 'destructive' });
    //   return;
    // }

    // Initialize Socket.IO connection
    console.log(`Attempting WebSocket connection to ${WEBSOCKET_URL}`);
    const newSocket = io(WEBSOCKET_URL, {
      path: WEBSOCKET_PATH,
      transports: ['websocket'], // Prefer WebSocket
      auth: { token }, // Send token if needed
      reconnectionAttempts: 5,
      reconnectionDelay: 3000,
    });

    newSocket.on('connect', () => {
      console.log('WebSocket connected:', newSocket.id);
      setIsConnected(true);
      setSocket(newSocket);
      // Optional: Subscribe to specific events if backend uses rooms
      // newSocket.emit('subscribe', { type: 'dashboard', userId });
      toast({ title: 'Real-time Connected', description: 'Receiving live updates.' });
    });

    newSocket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      setIsConnected(false);
      setSocket(null);
      if (reason !== 'io client disconnect') {
        toast({
          title: 'Real-time Disconnected',
          description: 'Attempting to reconnect...',
          variant: 'destructive',
        });
      }
    });

    newSocket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      setIsConnected(false);
      setSocket(null);
      toast({
        title: 'Real-time Connection Error',
        description: `Could not connect: ${error.message}`,
        variant: 'destructive',
      });
    });

    // --- Handle 'dashboard_update' event ---
    newSocket.on('dashboard_update', (data: { type: string, payload: { devices: Device[], dashboardMetrics: Partial<DashboardData> } }) => {
      console.log('Received dashboard_update via WebSocket:', data);
      if (data?.payload) {
        processDashboardUpdate(data.payload.devices || [], data.payload.dashboardMetrics || null);
      } else {
        console.warn('Received dashboard_update with invalid payload:', data);
      }
    });

    // --- Handle 'new_alert' event ---
    newSocket.on('new_alert', (data: { type: string, payload: Alert }) => {
      console.log('Received new_alert via WebSocket:', data);
      const newAlert = data?.payload;
      if (newAlert && newAlert.id) {
        setAlerts((prevAlerts) => {
          // Avoid adding duplicates
          if (prevAlerts.some((a) => a.id === newAlert.id)) {
            return prevAlerts;
          }
          const updatedAlerts = [newAlert, ...prevAlerts].slice(0, 100); // Add to top, limit size

          // Trigger pulse effect for new critical alerts
          if (newAlert.severity === 'critical' && !newAlert.acknowledged) {
             setNewCriticalAlerts((prevSet) => new Set([...prevSet, newAlert.id as string]));
             setTimeout(() => {
                setNewCriticalAlerts((prevSet) => {
                   const newSet = new Set(prevSet);
                   newSet.delete(newAlert.id as string);
                   return newSet;
                });
             }, 3000); // Clear pulse after 3 seconds

             toast({ // Also show a toast notification
                title: `🚨 Critical Alert: ${newAlert.title || newAlert.deviceId}`,
                description: newAlert.message || newAlert.description,
                variant: 'destructive',
                duration: 10000,
             });
          }
          return updatedAlerts;
        });
      } else {
         console.warn('Received new_alert with invalid payload:', data);
      }
    });

     // --- Handle 'alert_acknowledged' event (Optional but good practice) ---
     newSocket.on('alert_acknowledged', (data: { alertId: string; acknowledgedBy?: string }) => {
        console.log('Received acknowledgment update via WebSocket:', data);
        if (data?.alertId) {
            setAlerts((prevAlerts) =>
                prevAlerts.map((alert) =>
                    alert.id === data.alertId ? { ...alert, acknowledged: true } : alert
                )
            );
            // Maybe remove from newCriticalAlerts set if it was there
            setNewCriticalAlerts((prevSet) => {
                const newSet = new Set(prevSet);
                newSet.delete(data.alertId);
                return newSet;
            });
        }
     });


    // --- Cleanup WebSocket connection ---
    return () => {
      console.log('Cleaning up WebSocket connection.');
      newSocket.off('connect');
      newSocket.off('disconnect');
      newSocket.off('connect_error');
      newSocket.off('dashboard_update');
      newSocket.off('new_alert');
      newSocket.off('alert_acknowledged');
      newSocket.disconnect();
      setIsConnected(false);
      setSocket(null);
    };
  }, [userId, toast, processDashboardUpdate]); // Add dependencies

  // --- REMOVED: Polling useEffect ---
  // useEffect(() => {
  //   const interval = setInterval(() => {
  //     if (!loading) {
  //       loadData() // Now loadInitialData or let WebSocket handle updates
  //       setLastUpdate(Date.now())
  //     }
  //   }, 5000)
  //   return () => clearInterval(interval)
  // }, [loading, loadInitialData]) // Changed dependency

  // --- Manual Refresh Handler ---
  const handleRefresh = useCallback(() => {
    if (loading) return; // Prevent multiple refreshes
    console.log('Manual refresh triggered.');
    setLoading(true); // Show loading indicator during refresh
    loadInitialData()
      .then(() => {
        toast({
          title: 'Data Refreshed',
          description: 'Dashboard data has been updated.',
        });
      })
      .catch(() => { /* Error handled in loadInitialData */ })
      .finally(() => setLoading(false));
  }, [loadInitialData, toast, loading]);

  // --- Acknowledge Alert Handler (remains mostly the same, uses API/DB util) ---
  const handleAcknowledgeAlert = useCallback(async (alertId: string) => {
      // Optimistic UI update
      setAlerts((prevAlerts) =>
        prevAlerts.map((alert) =>
          alert.id === alertId ? { ...alert, acknowledged: true } : alert
        )
      );
       // Remove from pulsing set if present
       setNewCriticalAlerts((prevSet) => {
          const newSet = new Set(prevSet);
          newSet.delete(alertId);
          return newSet;
       });

    try {
      await acknowledgeAlert(alertId); // Call DB/API function
      toast({
        title: 'Alert Acknowledged',
        description: 'The alert has been marked as acknowledged.',
      });
      // Optional: If backend doesn't broadcast acknowledgments, emit it back
      // if (socket) {
      //   socket.emit('acknowledge_alert', { alertId, userId });
      // }
    } catch (error) {
      console.error('Error acknowledging alert:', error);
      toast({
        title: 'Error',
        description: 'Failed to acknowledge alert. Reverting UI.',
        variant: 'destructive',
      });
      // Revert optimistic update on error
      setAlerts((prevAlerts) =>
        prevAlerts.map((alert) =>
          alert.id === alertId ? { ...alert, acknowledged: false } : alert
        )
      );
    }
  }, [toast, userId, socket]); // Include socket if emitting

  // --- Export Handlers (remain the same) ---
  const handleExport = useCallback(async () => {
    // ... (implementation unchanged)
     try {
      setIsExporting(true);
      const csvData = [
        ['Device ID', 'Name', 'Type', 'Status', 'Value', 'Unit', 'Health Score', 'Efficiency Score', 'Location', 'Last Updated'],
        ...devices.map(device => [
          device.id,
          device.name,
          device.type,
          device.status,
          device.value?.toString() || 'N/A', // Handle potential undefined value
          device.unit,
          device.healthScore !== undefined ? (device.healthScore * 100).toFixed(1) + '%' : 'N/A', // Handle undefined
          device.efficiencyScore !== undefined ? (device.efficiencyScore * 100).toFixed(1) + '%' : 'N/A', // Handle undefined
          device.location,
          device.timestamp ? new Date(device.timestamp).toLocaleString() : 'N/A' // Handle undefined
        ])
      ];

      const csvContent = csvData.map(row => row.join(',')).join('\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' }); // Added charset
      const url = window.URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = url;
      a.download = `iot-dashboard-export-${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      toast({
        title: "Export Successful",
        description: "Device data has been exported to CSV."
      });
    } catch (error) {
      console.error('Error exporting data:', error);
      toast({
        title: "Export Failed",
        description: "Failed to export data.",
        variant: "destructive"
      });
    } finally {
      setIsExporting(false);
    }
  }, [devices, toast]);

  const handleGenerateReport = useCallback(async () => {
    // ... (implementation unchanged)
     try {
      setIsExporting(true);
      const reportData = {
        generatedAt: new Date().toISOString(),
        summary: {
          totalDevices: devices.length,
          activeDevices: devices.filter(d => d.status !== 'offline').length,
          systemHealth: dashboardData?.systemHealth || 0,
          efficiency: dashboardData?.efficiency || 0,
          energyUsage: dashboardData?.energyUsage || 0
        },
        devices: devices.map(device => ({
          ...device,
          healthScorePercent: device.healthScore !== undefined ? (device.healthScore * 100).toFixed(1) : 'N/A', // Handle undefined
          efficiencyScorePercent: device.efficiencyScore !== undefined ? (device.efficiencyScore * 100).toFixed(1) : 'N/A' // Handle undefined
        })),
        alerts: alerts.slice(0, 20), // Include recent alerts
        statusDistribution: dashboardData?.statusDistribution
      };

      const reportJson = JSON.stringify(reportData, null, 2);
      const blob = new Blob([reportJson], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = url;
      a.download = `iot-dashboard-report-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      toast({
        title: "Report Generated",
        description: "Comprehensive system report has been generated in JSON format."
      });
    } catch (error) {
      console.error('Error generating report:', error);
      toast({
        title: "Report Generation Failed",
        description: "Failed to generate report.",
        variant: "destructive"
      });
    } finally {
      setIsExporting(false);
    }
  }, [devices, alerts, dashboardData, toast]);
  // ------------------------------------

  return {
    devices,
    alerts,
    dashboardData,
    timeRange,
    loading,
    userId,
    lastUpdate,
    showThresholds,
    showEmailLogs,
    newCriticalAlerts,
    isExporting,
    isConnected, // Expose connection status
    setTimeRange,
    setShowThresholds,
    setShowEmailLogs,
    handleRefresh,
    handleAcknowledgeAlert,
    handleExport,
    handleGenerateReport,
  };
};