import { Card } from './ui/card'
import { cn } from '../utils/cn'
import { Alert } from '../types/digital-twin'
import { AlertTriangle, Info, XCircle, CheckCircle } from 'lucide-react'

interface AlertCardProps {
  alert: Alert
  onAcknowledge?: () => void
}

const severityConfig = {
  info: {
    icon: Info,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/20'
  },
  warning: {
    icon: AlertTriangle,
    color: 'text-yellow-500',
    bgColor: 'bg-yellow-500/10',
    borderColor: 'border-yellow-500/20'
  },
  critical: {
    icon: XCircle,
    color: 'text-red-500',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/20'
  }
}

export function AlertCard({ alert, onAcknowledge }: AlertCardProps) {
  const config = severityConfig[alert.severity]
  const Icon = config.icon
  
  return (
    <Card className={cn(
      "p-4 border-l-4 transition-all duration-300",
      config.borderColor,
      alert.acknowledged ? "opacity-60" : "hover:shadow-md"
    )}>
      <div className="flex items-start space-x-3">
        <div className={cn("p-2 rounded-full", config.bgColor)}>
          <Icon className={cn("h-4 w-4", config.color)} />
        </div>
        
        <div className="flex-1 space-y-1">
          <div className="flex items-start justify-between">
            <h4 className="font-medium text-foreground">{alert.title}</h4>
            {alert.acknowledged && (
              <CheckCircle className="h-4 w-4 text-green-500" />
            )}
          </div>
          
          <p className="text-sm text-muted-foreground">{alert.message}</p>
          
          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center space-x-4 text-xs text-muted-foreground">
              <span>{alert.deviceId}</span>
              <span>•</span>
              <span>{new Date(alert.timestamp).toLocaleString()}</span>
            </div>
            
            {!alert.acknowledged && onAcknowledge && (
              <button
                onClick={onAcknowledge}
                className="px-3 py-1 text-xs font-medium bg-primary text-primary-foreground rounded-md hover:bg-primary/80 transition-colors"
              >
                Acknowledge
              </button>
            )}
          </div>
        </div>
      </div>
    </Card>
  )
}