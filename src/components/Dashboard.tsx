import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { KPICard } from './KPICard'
import { DeviceCard } from './DeviceCard'
import { AlertCard } from './AlertCard'
import { ThresholdSettings } from './ThresholdSettings'
import { EmailLogsView } from './EmailLogsView'
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
  Mail
} from 'lucide-react'
import { TimeRange } from '../types/digital-twin'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { cn } from '../utils/cn'
import { Button } from './ui/button'
import { useDashboard } from '../hooks/useDashboard' // Import the new hook

export function Dashboard() {
  const {
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
    setTimeRange,
    setShowThresholds,
    setShowEmailLogs,
    handleRefresh,
    handleAcknowledgeAlert,
    handleExport,
    handleGenerateReport
  } = useDashboard() // Use the hook to get state and handlers

  const timeRangeOptions: { label: string; value: TimeRange }[] = [
    { label: '1H', value: '1h' },
    { label: '4H', value: '4h' },
    { label: '24H', value: '24h' },
    { label: '7D', value: '7d' },
    { label: '30D', value: '30d' }
  ]

  if (loading) {
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
      <div className="border-b border-border bg-card/50 backdrop-blur-sm">
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
                  <Settings className="h-4 w-4" />
                  Settings
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowEmailLogs(true)}
                >
                  <Mail className="h-4 w-4" />
                  Email Logs
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRefresh}
                >
                  <RefreshCw className="h-4 w-4" />
                  Refresh
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleExport}
                  disabled={isExporting}
                >
                  <Download className="h-4 w-4" />
                  Export
                </Button>
                
                <Button
                  size="sm"
                  onClick={handleGenerateReport}
                  disabled={isExporting}
                >
                  <FileText className="h-4 w-4" />
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

        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
          <KPICard
            title="System Health"
            value={`${dashboardData?.systemHealth || 0}%`}
            icon={Activity}
            variant="success"
            trend={{ value: "+2.1%", direction: "up" }}
            loading={!dashboardData}
          />
          
          <KPICard
            title="Active Devices"
            value={`${dashboardData?.activeDevices || 0}/${dashboardData?.totalDevices || 0}`}
            subtitle={`${dashboardData?.statusDistribution.offline || 0} offline`}
            icon={Cpu}
            variant="default"
            loading={!dashboardData}
          />
          
          <KPICard
            title="Energy Usage"
            value={`${dashboardData?.energyUsage || 0} kW`}
            subtitle={`$${dashboardData?.energyCost || 0}/hour`}
            icon={Zap}
            variant="warning"
            trend={{ value: "-0.8%", direction: "down" }}
            loading={!dashboardData}
          />
          
          <KPICard
            title="Efficiency"
            value={`${dashboardData?.efficiency || 0}%`}
            subtitle="Predicted 24h"
            icon={TrendingUp}
            variant="info"
            trend={{ value: "+1.2%", direction: "up" }}
            loading={!dashboardData}
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
                    <LineChart data={dashboardData?.performanceData || []}>
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