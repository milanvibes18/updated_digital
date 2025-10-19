// src/components/AlertCard.tsx
import { Card } from './ui/card';
import { cn } from '../utils/cn';
// --- UPDATED: Use Alert type from the central types file ---
import { Alert } from '../types/digital-twin';
import { AlertTriangle, Info, XCircle, CheckCircle } from 'lucide-react';

interface AlertCardProps {
  alert: Alert;
  // --- UPDATED: onAcknowledge now accepts the alert ID ---
  onAcknowledge?: (alertId: string) => void;
}

const severityConfig = {
  info: {
    icon: Info,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/20',
  },
  warning: {
    icon: AlertTriangle,
    color: 'text-yellow-500',
    bgColor: 'bg-yellow-500/10',
    borderColor: 'border-yellow-500/20',
  },
  critical: {
    icon: XCircle,
    color: 'text-red-500',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/20',
  },
};

export function AlertCard({ alert, onAcknowledge }: AlertCardProps) {
  // --- Fallback for potentially missing severity ---
  const config = severityConfig[alert.severity] || severityConfig.info;
  const Icon = config.icon;

  // --- NEW: Handler for the button click ---
  const handleAcknowledgeClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent card click if button is clicked
    if (onAcknowledge) {
      onAcknowledge(alert.id); // Pass the ID
    }
  };

  return (
    <Card
      className={cn(
        'p-4 border-l-4 transition-all duration-300',
        config.borderColor,
        alert.acknowledged
          ? 'opacity-60 cursor-default' // Make acknowledged less interactive
          : 'hover:shadow-md cursor-pointer' // Add pointer cursor for clarity
      )}
      // --- Optional: Add click handler for the whole card if needed ---
      // onClick={() => console.log('Card clicked:', alert.id)}
    >
      <div className="flex items-start space-x-3">
        <div className={cn('p-2 rounded-full', config.bgColor)}>
          <Icon className={cn('h-4 w-4', config.color)} />
        </div>

        <div className="flex-1 space-y-1">
          <div className="flex items-start justify-between">
            {/* --- Ensure title exists, provide fallback --- */}
            <h4 className="font-medium text-foreground">{alert.title || 'Untitled Alert'}</h4>
            {alert.acknowledged && (
              <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0 ml-2" />
            )}
          </div>

          {/* --- Ensure message exists, provide fallback --- */}
          <p className="text-sm text-muted-foreground">
            {alert.message || alert.description || 'No details provided.'}
          </p>

          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center space-x-2 text-xs text-muted-foreground overflow-hidden whitespace-nowrap">
              {/* --- Added safety check for deviceId --- */}
              <span>{alert.deviceId || 'Unknown Device'}</span>
              <span>•</span>
              {/* --- Added safety check for timestamp --- */}
              <span>
                {alert.timestamp
                  ? new Date(alert.timestamp).toLocaleString()
                  : 'No timestamp'}
              </span>
            </div>

            {!alert.acknowledged && onAcknowledge && (
              <button
                // --- UPDATED: Use the new handler ---
                onClick={handleAcknowledgeClick}
                className="ml-4 px-3 py-1 text-xs font-medium bg-primary text-primary-foreground rounded-md hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 transition-colors flex-shrink-0"
              >
                Acknowledge
              </button>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}