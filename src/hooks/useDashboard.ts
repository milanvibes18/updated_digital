import { useState, useEffect, useRef, useCallback } from 'react'
import { blink } from '../blink/client'
import { DataSimulator, defaultThresholds } from '../utils/data-simulator'
import { 
  getDevices, 
  getAlerts, 
  generateSampleData, 
  initializeDatabase,
  acknowledgeAlert
} from '../utils/db'
import { Device, Alert, DashboardData, TimeRange } from '../types/digital-twin'
import { useToast } from '../hooks/use-toast'

export const useDashboard = () => {
  const [devices, setDevices] = useState<Device[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [timeRange, setTimeRange] = useState<TimeRange>('24h')
  const [loading, setLoading] = useState(true)
  const [userId] = useState('demo-user')
  const [lastUpdate, setLastUpdate] = useState(Date.now())
  const [showThresholds, setShowThresholds] = useState(false)
  const [showEmailLogs, setShowEmailLogs] = useState(false)
  const [newCriticalAlerts, setNewCriticalAlerts] = useState<Set<string>>(new Set())
  const [isExporting, setIsExporting] = useState(false)
  
  const simulatorRef = useRef<DataSimulator | null>(null)
  const { toast } = useToast()

  const loadData = useCallback(async () => {
    try {
      const [devicesData, alertsData] = await Promise.all([
        getDevices(userId),
        getAlerts(userId, 10)
      ])
      
      // Check for new critical alerts and add pulse effect
      setAlerts(prevAlerts => {
        const previousCriticalAlerts = prevAlerts.filter(a => a.severity === 'critical' && !a.acknowledged)
        const currentCriticalAlerts = alertsData.filter(a => a.severity === 'critical' && !a.acknowledged)
        
        const newAlerts = currentCriticalAlerts.filter(current => 
          !previousCriticalAlerts.some(prev => prev.id === current.id)
        )
        
        if (newAlerts.length > 0) {
          const newAlertIds = new Set(newAlerts.map(alert => alert.id))
          setNewCriticalAlerts(newAlertIds as Set<string>)
          
          // Clear the pulse effect after 3 seconds
          setTimeout(() => {
            setNewCriticalAlerts(new Set<string>())
          }, 3000)
        }
        return alertsData
      })
      
      setDevices(devicesData)
      
      // Calculate dashboard metrics
      const totalDevices = devicesData.length
      const activeDevices = devicesData.filter(d => d.status !== 'offline').length
      const systemHealth = totalDevices > 0 ? devicesData.reduce((avg, device) => avg + device.healthScore, 0) / totalDevices * 100 : 0
      const efficiency = totalDevices > 0 ? devicesData.reduce((avg, device) => avg + device.efficiencyScore, 0) / totalDevices * 100 : 0
      const energyUsage = devicesData
        .filter(d => d.type === 'power_meter')
        .reduce((sum, device) => sum + device.value, 0)
      
      // Generate sample performance data with real-time variation
      const performanceData = Array.from({ length: 24 }, (_, i) => ({
        timestamp: `${String(i).padStart(2, '0')}:00`,
        systemHealth: Math.max(0, Math.min(100, systemHealth + Math.sin(i * 0.2) * 10 + Math.random() * 5)),
        efficiency: Math.max(0, Math.min(100, efficiency + Math.cos(i * 0.15) * 8 + Math.random() * 4)),
        energyUsage: Math.max(0, energyUsage + Math.sin(i * 0.1) * energyUsage * 0.2)
      }))
      
      setDashboardData({
        systemHealth: Math.round(systemHealth),
        activeDevices,
        totalDevices,
        efficiency: Math.round(efficiency),
        energyUsage: Math.round(energyUsage),
        energyCost: Math.round(energyUsage * 0.12), // $0.12 per kWh
        performanceData,
        statusDistribution: {
          normal: devicesData.filter(d => d.status === 'normal').length,
          warning: devicesData.filter(d => d.status === 'warning').length,
          critical: devicesData.filter(d => d.status === 'critical').length,
          offline: devicesData.filter(d => d.status === 'offline').length
        }
      })
    } catch (error) {
      console.error('Error loading data:', error)
      toast({
        title: "Error",
        description: "Failed to load dashboard data.",
        variant: "destructive"
      })
    }
  }, [userId, toast])

  const initializeApp = useCallback(async () => {
    try {
      setLoading(true)
      await initializeDatabase()
      
      const existingDevices = await getDevices(userId)
      if (existingDevices.length === 0) {
        await generateSampleData(userId)
      }
      
      const existingThresholds = await (blink.db as any).alertThresholds.list({ where: { userId } })
      if (existingThresholds.length === 0) {
        await Promise.all(
          defaultThresholds.map((threshold: any) => 
            (blink.db as any).alertThresholds.create({
              ...threshold,
              id: `THRESH_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              userId
            })
          )
        )
      }
      
      simulatorRef.current = new DataSimulator(userId)
      simulatorRef.current.start()
      
      await loadData()
    } catch (error) {
      console.error('Error initializing app:', error)
      toast({
        title: "Initialization Error",
        description: "Failed to initialize the application.",
        variant: "destructive"
      })
    } finally {
      setLoading(false)
    }
  }, [userId, loadData, toast])

  useEffect(() => {
    initializeApp()
    
    return () => {
      if (simulatorRef.current) {
        simulatorRef.current.stop()
      }
    }
  }, [initializeApp])

  useEffect(() => {
    const interval = setInterval(() => {
      if (!loading) {
        loadData()
        setLastUpdate(Date.now())
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [loading, loadData])

  const handleRefresh = useCallback(() => {
    loadData()
    setLastUpdate(Date.now())
    toast({
      title: "Data Refreshed",
      description: "Dashboard data has been updated."
    })
  }, [loadData, toast])

  const handleAcknowledgeAlert = useCallback(async (alertId: string) => {
    try {
      await acknowledgeAlert(alertId)
      setAlerts(prevAlerts => 
        prevAlerts.map(alert => 
          alert.id === alertId ? { ...alert, acknowledged: true } : alert
        )
      )
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
  }, [toast])

  const handleExport = useCallback(async () => {
    try {
      setIsExporting(true)
      const csvData = [
        ['Device ID', 'Name', 'Type', 'Status', 'Value', 'Unit', 'Health Score', 'Efficiency Score', 'Location', 'Last Updated'],
        ...devices.map(device => [
          device.id,
          device.name,
          device.type,
          device.status,
          device.value.toString(),
          device.unit,
          (device.healthScore * 100).toFixed(1) + '%',
          (device.efficiencyScore * 100).toFixed(1) + '%',
          device.location,
          new Date(device.timestamp).toLocaleString()
        ])
      ]
      
      const csvContent = csvData.map(row => row.join(',')).join('\n')
      const blob = new Blob([csvContent], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      
      const a = document.createElement('a')
      a.href = url
      a.download = `iot-dashboard-export-${new Date().toISOString().split('T')[0]}.csv`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
      
      toast({
        title: "Export Successful",
        description: "Device data has been exported to CSV."
      })
    } catch (error) {
      console.error('Error exporting data:', error)
      toast({
        title: "Export Failed",
        description: "Failed to export data.",
        variant: "destructive"
      })
    } finally {
      setIsExporting(false)
    }
  }, [devices, toast])

  const handleGenerateReport = useCallback(async () => {
    try {
      setIsExporting(true)
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
          healthScorePercent: (device.healthScore * 100).toFixed(1),
          efficiencyScorePercent: (device.efficiencyScore * 100).toFixed(1)
        })),
        alerts: alerts.slice(0, 20),
        statusDistribution: dashboardData?.statusDistribution
      }
      
      const reportJson = JSON.stringify(reportData, null, 2)
      const blob = new Blob([reportJson], { type: 'application/json' })
      const url = window.URL.createObjectURL(blob)
      
      const a = document.createElement('a')
      a.href = url
      a.download = `iot-dashboard-report-${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
      
      toast({
        title: "Report Generated",
        description: "Comprehensive system report has been generated."
      })
    } catch (error) {
      console.error('Error generating report:', error)
      toast({
        title: "Report Generation Failed",
        description: "Failed to generate report.",
        variant: "destructive"
      })
    } finally {
      setIsExporting(false)
    }
  }, [devices, alerts, dashboardData, toast])

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
    setTimeRange,
    setShowThresholds,
    setShowEmailLogs,
    handleRefresh,
    handleAcknowledgeAlert,
    handleExport,
    handleGenerateReport
  }
}