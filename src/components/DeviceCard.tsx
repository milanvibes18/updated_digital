// src/components/DeviceCard.tsx
import { useState, useEffect } from 'react'; // Import hooks
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Device } from '../types/digital-twin';
import { cn } from '../utils/cn';
import { Thermometer, Zap, Waves, Gauge, Activity, Power, WifiOff } from 'lucide-react';
import { Skeleton } from './ui/skeleton'; // Import Skeleton

interface DeviceCardProps {
  // --- UPDATED: Props are now optional for loading state ---
  device?: Device;
  loading?: boolean;
  onClick?: (deviceId: string) => void;
}

const deviceTypeConfig = {
  temperature_sensor: { icon: Thermometer, unit: '°C' },
  pressure_sensor: { icon: Gauge, unit: 'kPa' },
  vibration_sensor: { icon: Waves, unit: 'mm/s' },
  humidity_sensor: { icon: Activity, unit: '%' },
  power_meter: { icon: Zap, unit: 'kW' },
  default: { icon: Power, unit: '' },
};

const statusConfig = {
  normal: 'bg-green-500',
  warning: 'bg-yellow-500',
  critical: 'bg-red-500',
  offline: 'bg-gray-500',
};

// --- NEW: Skeleton Component for Loading State ---
function DeviceCardSkeleton() {
  return (
    <Card className="shadow-sm transition-shadow duration-300 animate-pulse">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-6 w-6 rounded-md" />
          <div className="flex items-center space-x-2">
            <Skeleton className="h-5 w-20 rounded-full" />
            <Skeleton className="h-3 w-3 rounded-full" />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Skeleton className="h-6 w-3/4 mb-1 rounded" />
        <Skeleton className="h-4 w-1/2 rounded" />

        <div className="mt-4 pt-4 border-t border-border/50">
          <div className="flex items-baseline justify-between">
            <Skeleton className="h-4 w-1/3 rounded" />
            <Skeleton className="h-6 w-1/4 rounded" />
          </div>
          <div className="flex items-baseline justify-between mt-2">
            <Skeleton className="h-4 w-1/4 rounded" />
            <Skeleton className="h-5 w-1/3 rounded" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function DeviceCard({ device, loading = false, onClick }: DeviceCardProps) {
  // --- NEW: State for real-time refresh indicator ---
  const [lastTimestamp, setLastTimestamp] = useState(device?.timestamp);
  const [isPulsing, setIsPulsing] = useState(false);

  // --- NEW: Effect to detect timestamp changes and trigger pulse ---
  useEffect(() => {
    if (device && device.timestamp !== lastTimestamp) {
      setIsPulsing(true);
      setLastTimestamp(device.timestamp);
      // Remove pulse animation after it finishes
      const timer = setTimeout(() => setIsPulsing(false), 2000); // 2s animation
      return () => clearTimeout(timer);
    }
  }, [device, device?.timestamp, lastTimestamp]);

  // --- NEW: Handle Loading State ---
  if (loading || !device) {
    return <DeviceCardSkeleton />;
  }

  // --- Existing Logic ---
  const config = deviceTypeConfig[device.type] || deviceTypeConfig.default;
  const Icon = device.status === 'offline' ? WifiOff : config.icon;
  const statusColor = statusConfig[device.status];

  const handleCardClick = () => {
    if (onClick) {
      onClick(device.id);
    }
  };

  return (
    <Card
      className={cn(
        'shadow-sm hover:shadow-lg transition-all duration-300',
        onClick && 'cursor-pointer'
      )}
      onClick={handleCardClick}
    >
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <Icon
            className={cn(
              'h-6 w-6 text-muted-foreground transition-colors',
              device.status === 'offline' && 'text-gray-500'
            )}
          />
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className="capitalize">
              {device.type.replace(/_/g, ' ')}
            </Badge>
            {/* --- NEW: Added pulse animation on update --- */}
            <div
              className={cn(
                'h-3 w-3 rounded-full',
                statusColor,
                isPulsing && 'animate-pulse' // Apply pulse
              )}
            />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <CardTitle className="text-lg font-semibold truncate" title={device.name}>
          {device.name}
        </CardTitle>
        <CardDescription className="text-sm truncate" title={device.id}>
          {device.id}
        </CardDescription>

        <div className="mt-4 pt-4 border-t border-border/50">
          <div className="flex items-baseline justify-between">
            <span className="text-sm text-muted-foreground">
              {device.status === 'offline' ? 'Last seen' : 'Current Value'}
            </span>
            {device.status !== 'offline' ? (
              // --- NEW: Added pulse animation on update ---
              <span
                className={cn(
                  'text-xl font-bold text-foreground',
                  isPulsing && 'animate-pulse' // Apply pulse
                )}
              >
                {device.value?.toFixed(2)}
                {config.unit}
              </span>
            ) : (
              <span className="text-sm text-muted-foreground">
                {new Date(device.timestamp).toLocaleString()}
              </span>
            )}
          </div>
          <div className="flex items-baseline justify-between mt-2">
            <span className="text-sm text-muted-foreground">Health</span>
            <span
              className={cn(
                'text-lg font-bold',
                device.healthScore >= 80
                  ? 'text-success'
                  : device.healthScore >= 60
                  ? 'text-warning'
                  : 'text-destructive',
                isPulsing && 'animate-pulse' // Apply pulse
              )}
            >
              {device.healthScore?.toFixed(0)}%
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}