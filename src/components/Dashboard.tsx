import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
// --- NEW: Import Tabs ---
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import { KPICard } from './KPICard'
import { DeviceCard } from './DeviceCard'
import { AlertCard } from './AlertCard'
import { ThresholdSettings } from './ThresholdSettings'
import { EmailLogsView } from './EmailLogsView'
// --- NEW: Import new views ---
import { HistoricalAnalyticsView } from './HistoricalAnalyticsView'
import { AssetManagementView } from './AssetManagementView'
import { AlertsHistoryView } from './AlertsHistoryView'
import { 
  Activity, 
  Cpu, 
  Zap, 
  TrendingUp, 
  Wifi, 
  RefreshCw,
  Download,
  FileText,
  Bell,
  Settings,
  Mail,
  // --- NEW: Icons for tabs ---
  History,
  Database,
  BellRing
} from 'lucide-react'
import { Device, Alert, DashboardData, TimeRange } from '../types/digital-twin'
import {
  getDashboardData,
  acknowledgeAlert,
  generateReport,
  exportData
} from '../utils/api' 
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { cn } from '../utils/cn'
import { useToast } from '../hooks/use-toast'
import { Button } from './ui/button'

export function Dashboard() {
  // State for data received from the API/WebSocket
  const [devices, setDevices] = useState<Device[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  
  // Existing state
  const [timeRange, setTimeRange] = useState<TimeRange>('24h')
  const [loading, setLoading] = useState(true)
  const [userId] = useState('demo-user')
  const [lastUpdate, setLastUpdate] = useState(Date.now())
  const [showThresholds, setShowThresholds] = useState(false)
  const [showEmailLogs, setShowEmailLogs] = useState(false)
  const [newCriticalAlerts, setNewCriticalAlerts] = useState<Set<string>>(new Set())
  const [isExporting, setIsExporting] = useState(false)
  
  // --- NEW: State for active tab ---
  const [activeView, setActiveView] = useState('realtime')

  const { toast } = useToast()

  const timeRangeOptions: { label: string; value: TimeRange }[] = [
    { label: '1H', value: '1h' },
    { label: '4H', value: '4h' },
    { label: '24H', value: '24h' },
    { label: '7D', value: '7d' },
    { label: '30D', value: '30d' }
  ]

  // --- WebSocket and Initial Load Effect ---
  useEffect(() => {
    // Initial data load
    loadData()
    
    // WebSocket Event Listeners
    const handleSocketUpdate = (event: Event) => {
      const customEvent = event as CustomEvent<DashboardData>
      console.log('Socket data received:', customEvent.detail)
      setDashboardData(customEvent.detail) // Update dashboard with live data
      setDevices(customEvent.detail.devices || [])
      setLastUpdate(Date.now())
    }

    const handleSocketAlert = (event: Event) => {
      const customEvent = event as CustomEvent<Alert>
      console.log('Socket alert received:', customEvent.detail)
      // Add new alert to the top of the list
      setAlerts(prevAlerts => [customEvent.detail, ...prevAlerts])
      
      // Add pulse effect logic
      if (customEvent.detail.severity === 'critical' && !customEvent.detail.acknowledged) {
        setNewCriticalAlerts(prev => new Set(prev).add(customEvent.detail.id))
        
        // Clear the pulse effect after 3 seconds
        setTimeout(() => {
          setNewCriticalAlerts(prev => {
            const newSet = new Set(prev)
            newSet.delete(customEvent.detail.id)
            return newSet
          })
        }, 3000)
      }
    }

    document.addEventListener('socketDataUpdate', handleSocketUpdate as EventListener)
    document.addEventListener('socketAlertUpdate', handleSocketAlert as EventListener)

    return () => {
      document.removeEventListener('socketDataUpdate', handleSocketUpdate as EventListener)
      document.removeEventListener('socketAlertUpdate', handleSocketAlert as EventListener)
    }
  }, [])

  // Fallback refresh interval
  useEffect(() => {
    const interval = setInterval(() => {
      if (!loading) {
        setLastUpdate(Date.now())
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [loading])

  // --- UPDATED: Load data based on time range ---
  // This effect re-fetches data when the timeRange state changes.
  useEffect(() => {
    loadData()
  }, [timeRange, userId])

  const loadData = async () => {
    try {
      setLoading(true)
      
      // Initial load fetches all necessary dashboard data from the API
      const data = await getDashboardData(userId, timeRange)
      
      setDevices(data.devices)
      setAlerts(data.alerts)
      setDashboardData(data)
    } catch (error) {
      console.error('Error loading data:', error)
      toast({
        title: "Connection Error",
        description: "Failed to load dashboard data from API.",
        variant: "destructive"
      })
    } finally {
      setLoading(false)
    }
  }
  
  const handleRefresh = () => {
    loadData() // Force a refresh by calling the API
    setLastUpdate(Date.now())
    toast({
      title: "Data Refreshed",
      description: "Dashboard data has been updated from the server."
    })
  }

  const handleAcknowledgeAlert = async (alertId: string) => {
    try {
      await acknowledgeAlert(alertId) // Use API function
      setAlerts(alerts.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      ))
      toast({
        title: "Alert Acknowledged",
        description: "The alert has been marked as acknowledged."
      })
    } catch (error) {
      console.error('Error acknowledging alert:', error)
      toast({
        title: "Error",
        description: "Failed to acknowledge alert.",
        variant: "destructive"
      })
    }
  }

  // --- Handle Export (CSV) with API ---
  const handleExport = async () => {
    setIsExporting(true)
    try {
      const result = await exportData('csv', userId)
      const a = document.createElement('a')
      a.href = result.export_path // Assumes Flask serves from /exports/
      a.download = result.filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      
      toast({ title: 'Export Successful', description: 'Device data exported to CSV.' })
    } catch (error: any) {
      console.error('Error exporting data:', error)
      toast({
        title: 'Export Failed',
        description: error.message || 'An error occurred during export.',
        variant: 'destructive'
      })
    } finally {
      setIsExporting(false)
    }
  }

  // --- Handle Report Generation (JSON/PDF) with API ---
  const handleGenerateReport = async () => {
    setIsExporting(true)
    try {
      const result = await generateReport(userId)
      window.open(result.report_path, '_blank')
      
      toast({ title: 'Report Generated', description: 'Comprehensive system report generated.' })
    } catch (error: any) {
      console.error('Error generating report:', error)
      toast({
        title: 'Report Generation Failed',
        description: error.message || 'An error occurred during report generation.',
        variant: 'destructive'
      })
    } finally {
      setIsExporting(false)
    }
  }

  // Ensure dashboardData is not null before proceeding to render content that depends on it
  const displayData = dashboardData || {
    systemHealth: 0,
    activeDevices: 0,
    totalDevices: 0,
    efficiency: 0,
    energyUsage: 0,
    energyCost: 0,
    performanceData: [],
    statusDistribution: { normal: 0, warning: 0, critical: 0, offline: 0 }
  }

  if (loading && !dashboardData) { // Only show full-page load on initial mount
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex items-center space-x-2">
          <RefreshCw className="h-6 w-6 animate-spin" />
          <span>Loading Dashboard...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold flex items-center gap-3">
                <Activity className="h-8 w-8 text-primary" />
                Digital Twin Dashboard
              </h1>
              <p className="text-muted-foreground mt-1">
                Real-time Industrial IoT Monitoring & Analytics
              </p>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <Wifi className="h-4 w-4 text-green-500" />
                <span className="text-sm text-muted-foreground">Connected</span>
                <span className="text-xs text-muted-foreground/70 ml-2">
                  Last update: {new Date(lastUpdate).toLocaleTimeString()}
                </span>
              </div>
              
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowThresholds(true)}
                >
                  <Settings className="h-4 w-4 mr-2" />
                  Settings
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowEmailLogs(true)}
                >
                  <Mail className="h-4 w-4 mr-2" />
                  Email Logs
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRefresh}
                  disabled={loading}
                >
                  <RefreshCw className={cn("h-4 w-4 mr-2", loading && "animate-spin")} />
                  Refresh
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleExport}
                  disabled={isExporting}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </Button>
                
                <Button
                  size="sm"
                  onClick={handleGenerateReport}
                  disabled={isExporting}
                >
                  <FileText className="h-4 w-4 mr-2" />
                  Report
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Time Range Selector */}
        <div className="mb-8">
          <div className="flex items-center gap-2">
            {timeRangeOptions.map(option => (
              <button
                key={option.value}
                onClick={() => setTimeRange(option.value)}
                className={cn(
                  "px-3 py-1 rounded-md text-sm font-medium transition-colors",
                  timeRange === option.value 
                    ? "bg-primary text-primary-foreground" 
                    : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* --- NEW: Tabbed Interface --- */}
        <Tabs value={activeView} onValueChange={setActiveView} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="realtime">
              <Activity className="h-4 w-4 mr-2" />
              Real-time
            </TabsTrigger>
            <TabsTrigger value="historical">
              <History className="h-4 w-4 mr-2" />
              Historical Analytics
            </TabsTrigger>
            <TabsTrigger value="assets">
              <Database className="h-4 w-4 mr-2" />
              Asset Management
            </TabsTrigger>
            <TabsTrigger value="alerts">
              <BellRing className="h-4 w-4 mr-2" />
              All Alerts
            </TabsTrigger>
          </TabsList>

          {/* --- UPDATED: Real-time Tab --- */}
          <TabsContent value="realtime">
            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
              <KPICard
                title="System Health"
                value={`${displayData.systemHealth || 0}%`}
                icon={Activity}
                variant="success"
                trend={{ value: "+2.1%", direction: "up" }}
                loading={loading}
              />
              
              <KPICard
                title="Active Devices"
                value={`${displayData.activeDevices || 0}/${displayData.totalDevices || 0}`}
                subtitle={`${displayData.statusDistribution.offline || 0} offline`}
                icon={Cpu}
                variant="default"
                loading={loading}
              />
              
              <KPICard
                title="Energy Usage"
                value={`${displayData.energyUsage || 0} kW`}
                subtitle={`$${displayData.energyCost || 0}/hour`}
                icon={Zap}
                variant="warning"
                trend={{ value: "-0.8%", direction: "down" }}
                loading={loading}
              />
              
              <KPICard
                title="Efficiency"
                value={`${displayData.efficiency || 0}%`}
                subtitle="Predicted 24h"
                icon={TrendingUp}
                variant="info"
                trend={{ value: "+1.2%", direction: "up" }}
                loading={loading}
              />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
              {/* Performance Chart */}
              <div className="xl:col-span-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Performance Metrics</CardTitle>
                    <CardDescription>Real-time system performance over time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[400px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={displayData.performanceData || []}>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                          <XAxis dataKey="timestamp" stroke="hsl(var(--muted-foreground))" />
                          <YAxis stroke="hsl(var(--muted-foreground))" />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: 'hsl(var(--card))', 
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '8px'
                            }} 
                          />
                          <Line 
                            type="monotone" 
                            dataKey="systemHealth" 
                            stroke="hsl(var(--success))" 
                            strokeWidth={2}
                            dot={false}
                            name="System Health %"
                          />
                          <Line 
                            type="monotone" 
                            dataKey="efficiency" 
                            stroke="hsl(var(--primary))" 
                            strokeWidth={2}
                            dot={false}
                            name="Efficiency %"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Critical Alerts */}
              <div>
                <Card className={cn(
                  "transition-all duration-300",
                  newCriticalAlerts.size > 0 && "ring-2 ring-red-500 shadow-red-500/20 shadow-lg animate-pulse"
                )}>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Bell className={cn(
                        "h-5 w-5",
                        newCriticalAlerts.size > 0 && "text-red-500 animate-pulse"
                      )} />
                      Critical Alerts
                      <span className={cn(
                        "ml-auto text-white text-xs px-2 py-1 rounded-full transition-colors",
                        newCriticalAlerts.size > 0 
                          ? "bg-red-600 animate-pulse" 
                          : "bg-red-500"
                      )}>
                        {alerts.filter(a => a.severity === 'critical' && !a.acknowledged).length}
                      </span>
                    </CardTitle>
                    <CardDescription>Requires immediate attention</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {alerts.filter(a => a.severity === 'critical').slice(0, 3).map(alert => (
                      <div
                        key={alert.id}
                        className={cn(
                          "transition-all duration-300",
                          newCriticalAlerts.has(alert.id) && "ring-2 ring-red-400 shadow-md animate-pulse"
                        )}
                      >
                        <AlertCard
                          alert={alert}
                          onAcknowledge={() => handleAcknowledgeAlert(alert.id)}
                        />
                      </div>
                    ))}
                    {alerts.filter(a => a.severity === 'critical').length === 0 && (
                      <p className="text-muted-foreground text-center py-8">
                        No critical alerts
                      </p>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Devices Grid */}
            <div className="mt-8">
              <Card>
                <CardHeader>
                  <CardTitle>Device Status</CardTitle>
                  <CardDescription>Real-time monitoring of all connected devices</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4">
                    {devices.map(device => (
                      <DeviceCard
                        key={device.id}
                        device={device}
                        onClick={() => console.log('Device clicked:', device.id)}
                      />
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* --- NEW: Historical Analytics Tab --- */}
          <TabsContent value="historical">
            <HistoricalAnalyticsView userId={userId} timeRange={timeRange} />
          </TabsContent>

          {/* --- NEW: Asset Management Tab --- */}
          <TabsContent value="assets">
            <AssetManagementView userId={userId} />
          </TabsContent>

          {/* --- NEW: All Alerts Tab --- */}
          <TabsContent value="alerts">
            <AlertsHistoryView 
              userId={userId} 
              initialAlerts={alerts} 
              onAcknowledge={handleAcknowledgeAlert}
            />
          </TabsContent>
        </Tabs>
      </div>

      {/* Settings Modal */}
      {showThresholds && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <ThresholdSettings 
            userId={userId} 
            onClose={() => setShowThresholds(false)} 
          />
        </div>
      )}

      {/* Email Logs Modal */}
      {showEmailLogs && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <EmailLogsView 
            userId={userId} 
            onClose={() => setShowEmailLogs(false)} 
          />
        </div>
      )}
    </div>
  )
}