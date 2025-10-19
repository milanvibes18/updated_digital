import { blink } from '../blink/client'
import type { Device, Alert, SystemMetrics, MaintenanceTask, Recommendation } from '../types/digital-twin'

// Initialize database tables
export const initializeDatabase = async () => {
  try {
    await blink.db.devices.create({
      id: 'init',
      name: 'init',
      type: 'temperature_sensor',
      status: 'normal',
      value: 0,
      unit: '°C',
      healthScore: 1.0,
      efficiencyScore: 1.0,
      location: 'init',
      timestamp: new Date().toISOString(),
      userId: 'system'
    })
    
    await blink.db.devices.delete('init')
    console.log('Database tables initialized successfully')
  } catch (error) {
    console.error('Error initializing database:', error)
  }
}

// Generate sample data
export const generateSampleData = async (userId: string) => {
  const deviceTypes = ['temperature_sensor', 'pressure_sensor', 'vibration_sensor', 'humidity_sensor', 'power_meter'] as const
  const locations = ['Factory Floor A', 'Factory Floor B', 'Warehouse', 'Quality Lab']
  const statuses = ['normal', 'warning', 'critical', 'offline'] as const
  
  // Create sample devices
  const devices = Array.from({ length: 15 }, (_, i) => {
    const deviceType = deviceTypes[i % deviceTypes.length]
    const statusRandom = Math.random()
    
    let status: Device['status'] = 'normal'
    let healthScore = Math.random() * 0.2 + 0.8
    
    if (statusRandom > 0.1 && statusRandom <= 0.15) {
      status = 'warning'
      healthScore = Math.random() * 0.3 + 0.5
    } else if (statusRandom > 0.05 && statusRandom <= 0.1) {
      status = 'critical'
      healthScore = Math.random() * 0.4 + 0.1
    } else if (statusRandom <= 0.05) {
      status = 'offline'
      healthScore = 0
    }
    
    return {
      id: `DEVICE_${String(i + 1).padStart(3, '0')}`,
      name: `${deviceType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} ${String(i + 1).padStart(3, '0')}`,
      type: deviceType,
      status,
      value: Math.round((Math.random() * 90 + 10) * 100) / 100,
      unit: getUnitForType(deviceType),
      healthScore,
      efficiencyScore: Math.random() * 0.3 + 0.7,
      location: locations[i % locations.length],
      timestamp: new Date().toISOString(),
      userId
    }
  })
  
  // Create sample alerts
  const alertTypes = [
    { title: 'Temperature Anomaly', message: 'Temperature threshold exceeded', severity: 'warning' as const },
    { title: 'Pressure Critical', message: 'Pressure levels critical', severity: 'critical' as const },
    { title: 'Device Offline', message: 'Device connection lost', severity: 'critical' as const },
    { title: 'High Vibration', message: 'Vibration levels above normal', severity: 'warning' as const },
    { title: 'Maintenance Due', message: 'Scheduled maintenance required', severity: 'info' as const }
  ]
  
  const alerts = Array.from({ length: 8 }, (_, i) => {
    const alertType = alertTypes[i % alertTypes.length]
    const deviceId = devices[i % devices.length].id
    
    return {
      id: `ALERT_${String(i + 1).padStart(3, '0')}`,
      title: alertType.title,
      message: `${alertType.message} on ${deviceId}`,
      severity: alertType.severity,
      deviceId,
      timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString(),
      acknowledged: Math.random() > 0.3,
      userId
    }
  })
  
  // Create sample maintenance tasks
  const maintenanceTasks = Array.from({ length: 5 }, (_, i) => ({
    id: `MAINT_${String(i + 1).padStart(3, '0')}`,
    deviceId: devices[i % devices.length].id,
    title: `Routine Maintenance ${i + 1}`,
    description: `Scheduled maintenance for ${devices[i % devices.length].name}`,
    priority: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as MaintenanceTask['priority'],
    dueDate: new Date(Date.now() + Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
    status: 'pending' as const,
    estimatedDuration: Math.floor(Math.random() * 240 + 60),
    userId
  }))
  
  // Create sample recommendations
  const recommendations = [
    {
      id: 'REC_001',
      type: 'maintenance' as const,
      priority: 'high' as const,
      title: 'Schedule Preventive Maintenance',
      description: 'Device showing increased wear patterns. Recommend maintenance within 48 hours.',
      estimatedImpact: 'Prevent potential downtime of 4-6 hours',
      confidence: 0.92,
      deviceId: devices[2].id,
      userId,
      createdAt: new Date().toISOString()
    },
    {
      id: 'REC_002',
      type: 'optimization' as const,
      priority: 'medium' as const,
      title: 'Optimize Operating Parameters',
      description: 'Adjust setpoints to improve energy efficiency by 8-12%',
      estimatedImpact: 'Reduce energy consumption and costs',
      confidence: 0.85,
      userId,
      createdAt: new Date().toISOString()
    }
  ]
  
  try {
    // Insert all sample data
    await Promise.all([
      ...devices.map(device => db.devices.create(device)),
      ...alerts.map(alert => db.alerts.create(alert)),
      ...maintenanceTasks.map(task => db.maintenanceTasks.create(task)),
      ...recommendations.map(rec => db.recommendations.create(rec))
    ])
    
    console.log('Sample data generated successfully')
  } catch (error) {
    console.error('Error generating sample data:', error)
  }
}

function getUnitForType(deviceType: Device['type']): string {
  const units = {
    temperature_sensor: '°C',
    pressure_sensor: 'hPa',
    vibration_sensor: 'mm/s',
    humidity_sensor: '%RH',
    power_meter: 'W'
  }
  return units[deviceType]
}

// Type-safe database access helpers
const db = {
  devices: (blink.db as any).devices,
  alerts: (blink.db as any).alerts,
  maintenanceTasks: (blink.db as any).maintenanceTasks,
  recommendations: (blink.db as any).recommendations,
  alertThresholds: (blink.db as any).alertThresholds,
  emailAlerts: (blink.db as any).emailAlerts
}

// Data fetching functions
export const getDevices = (userId: string) => 
  db.devices.list({ where: { userId }, orderBy: { timestamp: 'desc' } })

export const getAlerts = (userId: string, limit = 10) => 
  db.alerts.list({ where: { userId }, orderBy: { timestamp: 'desc' }, limit })

export const getMaintenanceTasks = (userId: string) => 
  db.maintenanceTasks.list({ where: { userId }, orderBy: { dueDate: 'asc' } })

export const getRecommendations = (userId: string) => 
  db.recommendations.list({ where: { userId }, orderBy: { createdAt: 'desc' } })

// Update functions
export const updateDeviceStatus = (deviceId: string, status: Device['status']) =>
  db.devices.update(deviceId, { status, timestamp: new Date().toISOString() })

export const acknowledgeAlert = (alertId: string) =>
  db.alerts.update(alertId, { acknowledged: true })

export const completeMaintenanceTask = (taskId: string) =>
  db.maintenanceTasks.update(taskId, { status: 'completed' })