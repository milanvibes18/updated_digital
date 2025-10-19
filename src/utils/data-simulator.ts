import { Device, Alert, AlertThreshold } from '../types/digital-twin'
import { blink } from '../blink/client'

export class DataSimulator {
  private intervalId: number | null = null
  private alertCheckIntervalId: number | null = null
  private alertHistory = new Set<string>() // For deduplication
  private userId: string

  constructor(userId: string) {
    this.userId = userId
  }

  start() {
    // Initial data update
    this.updateDeviceData()
    
    // Update simulated data every 5 seconds
    this.intervalId = window.setInterval(() => {
      this.updateDeviceData()
    }, 5000)

    // Check for alerts every 3 seconds
    this.alertCheckIntervalId = window.setInterval(() => {
      this.checkAlertThresholds()
    }, 3000)
  }

  stop() {
    if (this.intervalId) {
      window.clearInterval(this.intervalId)
      this.intervalId = null
    }
    if (this.alertCheckIntervalId) {
      window.clearInterval(this.alertCheckIntervalId)
      this.alertCheckIntervalId = null
    }
  }

  private async updateDeviceData() {
    try {
      const devices = await (blink.db as any).devices.list({ 
        where: { userId: this.userId }, 
        orderBy: { timestamp: 'desc' } 
      })

      const updates = devices.map(device => {
        if (device.status === 'offline') return null

        // Generate realistic sensor readings with some variation
        let newValue = device.value
        const variation = this.getVariationForDeviceType(device.type)
        newValue = device.value + (Math.random() - 0.5) * variation
        newValue = Math.max(0, Math.round(newValue * 100) / 100)

        // Occasionally introduce anomalies
        if (Math.random() < 0.05) { // 5% chance of anomaly
          newValue = newValue * (1 + (Math.random() - 0.5) * 0.5) // +/- 25%
        }

        // Update health and efficiency scores based on value and status
        let healthScore = device.healthScore
        let efficiencyScore = device.efficiencyScore
        let newStatus = device.status

        // Determine status based on value thresholds
        if (device.type === 'temperature_sensor' && newValue > 45) {
          newStatus = 'critical'
        } else if (device.type === 'temperature_sensor' && newValue > 35) {
          newStatus = 'warning'
        } else if (device.type === 'vibration_sensor' && newValue > 25) {
          newStatus = 'critical'
        } else if (device.type === 'vibration_sensor' && newValue > 15) {
          newStatus = 'warning'
        } else if (device.type === 'pressure_sensor' && newValue > 1200) {
          newStatus = 'critical'
        } else if (device.type === 'power_meter' && newValue > 1000) {
          newStatus = 'warning'
        } else if (Math.random() < 0.95) { // 95% chance to be normal if not threshold exceeded
          newStatus = 'normal'
        }

        if (newStatus === 'critical') {
          healthScore = Math.max(0.1, healthScore - 0.02)
          efficiencyScore = Math.max(0.3, efficiencyScore - 0.01)
        } else if (newStatus === 'warning') {
          healthScore = healthScore + (Math.random() - 0.6) * 0.02
          efficiencyScore = efficiencyScore + (Math.random() - 0.5) * 0.01
        } else {
          healthScore = Math.min(1.0, healthScore + 0.002)
          efficiencyScore = Math.min(1.0, efficiencyScore + 0.001)
        }

        return (blink.db as any).devices.update(device.id, {
          value: newValue,
          status: newStatus,
          healthScore: Math.round(healthScore * 1000) / 1000,
          efficiencyScore: Math.round(efficiencyScore * 1000) / 1000,
          timestamp: new Date().toISOString()
        })
      }).filter(Boolean)

      await Promise.all(updates)
    } catch (error) {
      console.error('Error updating simulated data:', error)
    }
  }

  private getVariationForDeviceType(type: Device['type']): number {
    const variations = {
      temperature_sensor: 2.0, // ±1°C
      pressure_sensor: 10.0, // ±5hPa
      vibration_sensor: 1.0, // ±0.5mm/s
      humidity_sensor: 3.0, // ±1.5%RH
      power_meter: 50.0 // ±25W
    }
    return variations[type] || 1.0
  }

  private async checkAlertThresholds() {
    try {
      const [devices, thresholds] = await Promise.all([
        (blink.db as any).devices.list({ where: { userId: this.userId } }),
        (blink.db as any).alertThresholds.list({ 
          where: { userId: this.userId, enabled: "1" } 
        })
      ])

      const newAlerts: Alert[] = []

      for (const device of devices) {
        const deviceThresholds = thresholds.filter(t => 
          t.deviceType === device.type || t.deviceType === 'all'
        )

        for (const threshold of deviceThresholds) {
          if (this.checkThreshold(device.value, threshold)) {
            const alertKey = `${device.id}-${threshold.id}-${Date.now()}`
            
            // Deduplication: only create alert if we haven't seen this combination recently
            const dedupeKey = `${device.id}-${threshold.metricType}-${threshold.severity}`
            if (this.alertHistory.has(dedupeKey)) continue

            this.alertHistory.add(dedupeKey)
            // Remove from history after 5 minutes
            setTimeout(() => this.alertHistory.delete(dedupeKey), 5 * 60 * 1000)

            const alert: Alert = {
              id: `ALERT_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              title: this.getAlertTitle(threshold),
              message: `${threshold.metricType} ${threshold.operator} ${threshold.value}${device.unit} on ${device.name} (current: ${device.value}${device.unit})`,
              severity: threshold.severity,
              deviceId: device.id,
              timestamp: new Date().toISOString(),
              acknowledged: false,
              userId: this.userId
            }

            newAlerts.push(alert)
          }
        }
      }

      // Create all new alerts
      if (newAlerts.length > 0) {
        await Promise.all(newAlerts.map(alert => 
          (blink.db as any).alerts.create(alert)
        ))

        // Trigger email alerts for critical ones
        const criticalAlerts = newAlerts.filter(a => a.severity === 'critical')
        for (const alert of criticalAlerts) {
          this.sendEmailAlert(alert)
        }
      }
    } catch (error) {
      console.error('Error checking alert thresholds:', error)
    }
  }

  private checkThreshold(value: number, threshold: AlertThreshold): boolean {
    switch (threshold.operator) {
      case 'gt': return value > threshold.value
      case 'gte': return value >= threshold.value
      case 'lt': return value < threshold.value
      case 'lte': return value <= threshold.value
      case 'eq': return Math.abs(value - threshold.value) < 0.01
      default: return false
    }
  }

  private getAlertTitle(threshold: AlertThreshold): string {
    const titles = {
      'temperature': 'Temperature Alert',
      'pressure': 'Pressure Alert', 
      'vibration': 'Vibration Alert',
      'humidity': 'Humidity Alert',
      'power': 'Power Alert'
    }
    return titles[threshold.metricType as keyof typeof titles] || 'Device Alert'
  }

  private async sendEmailAlert(alert: Alert) {
    try {
      const emailAlert = {
        id: `EMAIL_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        alertId: alert.id,
        recipient: 'devicelogin87@gmail.com',
        subject: `🚨 CRITICAL ALERT: ${alert.title}`,
        body: `
          <h2>Critical Alert Notification</h2>
          <p><strong>Device:</strong> ${alert.deviceId}</p>
          <p><strong>Alert:</strong> ${alert.title}</p>
          <p><strong>Message:</strong> ${alert.message}</p>
          <p><strong>Time:</strong> ${new Date(alert.timestamp).toLocaleString()}</p>
          <p><strong>Severity:</strong> ${alert.severity.toUpperCase()}</p>
          
          <p>Please take immediate action to address this issue.</p>
          
          <hr>
          <p><small>Sent from Digital Twin IoT Platform</small></p>
        `,
        sentAt: new Date().toISOString(),
        status: 'pending' as const,
        userId: this.userId
      }

      await (blink.db as any).emailAlerts.create(emailAlert)
      
      // Simulate email sending (in real implementation, this would use Gmail API)
      setTimeout(async () => {
        await (blink.db as any).emailAlerts.update(emailAlert.id, {
          status: Math.random() > 0.1 ? 'sent' : 'failed',
          ...(Math.random() <= 0.1 && { errorMessage: 'Network timeout' })
        })
      }, 2000)

    } catch (error) {
      console.error('Error sending email alert:', error)
    }
  }
}

// Default thresholds
export const defaultThresholds: Omit<AlertThreshold, 'id' | 'userId'>[] = [
  {
    deviceType: 'temperature_sensor',
    metricType: 'temperature',
    operator: 'gt',
    value: 45,
    severity: 'critical',
    enabled: true
  },
  {
    deviceType: 'temperature_sensor', 
    metricType: 'temperature',
    operator: 'gt',
    value: 35,
    severity: 'warning',
    enabled: true
  },
  {
    deviceType: 'pressure_sensor',
    metricType: 'pressure',
    operator: 'gt',
    value: 1200,
    severity: 'critical', 
    enabled: true
  },
  {
    deviceType: 'vibration_sensor',
    metricType: 'vibration',
    operator: 'gt',
    value: 15,
    severity: 'warning',
    enabled: true
  },
  {
    deviceType: 'vibration_sensor',
    metricType: 'vibration',
    operator: 'gt', 
    value: 25,
    severity: 'critical',
    enabled: true
  },
  {
    deviceType: 'power_meter',
    metricType: 'power',
    operator: 'gt',
    value: 1000,
    severity: 'warning',
    enabled: true
  }
]