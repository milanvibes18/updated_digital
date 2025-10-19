import { Card } from './ui/card'
import { cn } from '../utils/cn'
import { Device } from '../types/digital-twin'
import { Thermometer, Gauge, Activity, Droplet, Zap } from 'lucide-react'

interface DeviceCardProps {
  device: Device
  onClick?: () => void
}

const statusColors = {
  normal: 'bg-green-500/20 text-green-500 border-green-500/30',
  warning: 'bg-yellow-500/20 text-yellow-500 border-yellow-500/30',
  critical: 'bg-red-500/20 text-red-500 border-red-500/30',
  offline: 'bg-gray-500/20 text-gray-500 border-gray-500/30'
}

const deviceIcons = {
  temperature_sensor: Thermometer,
  pressure_sensor: Gauge,
  vibration_sensor: Activity,
  humidity_sensor: Droplet,
  power_meter: Zap
}

export function DeviceCard({ device, onClick }: DeviceCardProps) {
  const Icon = deviceIcons[device.type]
  const healthPercentage = Math.round(device.healthScore * 100)
  
  return (
    <Card 
      className={cn(
        "p-4 cursor-pointer transition-all duration-300 hover:shadow-lg hover:-translate-y-1",
        onClick && "hover:border-primary/50"
      )}
      onClick={onClick}
    >
      <div className="space-y-4">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <h3 className="font-semibold text-foreground">{device.name}</h3>
            <p className="text-sm text-muted-foreground">{device.location}</p>
          </div>
          <Icon className="h-6 w-6 text-muted-foreground" />
        </div>
        
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-2xl font-bold text-foreground">
              {device.value} {device.unit}
            </span>
            <span className={cn(
              "px-2 py-1 rounded-full text-xs font-medium border",
              statusColors[device.status]
            )}>
              {device.status}
            </span>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Health</span>
              <span className="font-medium">{healthPercentage}%</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2">
              <div 
                className={cn(
                  "h-2 rounded-full transition-all duration-500",
                  healthPercentage >= 80 && "bg-green-500",
                  healthPercentage >= 60 && healthPercentage < 80 && "bg-yellow-500",
                  healthPercentage < 60 && "bg-red-500"
                )}
                style={{ width: `${healthPercentage}%` }}
              />
            </div>
          </div>
          
          <p className="text-xs text-muted-foreground">
            Updated {new Date(device.timestamp).toLocaleTimeString()}
          </p>
        </div>
      </div>
    </Card>
  )
}