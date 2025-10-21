// src/components/DeviceCard.tsx
import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Device } from '../types/digital-twin';
import { cn } from '../utils/cn';
import {
    Thermometer, Zap, Waves, Gauge, Activity, Power, WifiOff,
    TrendingUp, TrendingDown, Minus, // Added trend icons
    Info // Added Info icon for details/comparison placeholder
} from 'lucide-react';
import { Skeleton } from './ui/skeleton';
import { Button } from './ui/button'; // Added for placeholder
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip'; // Added for tooltips

interface DeviceCardProps {
  device?: Device;
  loading?: boolean;
  onClick?: (deviceId: string) => void;
  // --- NEW ---
  trend?: 'up' | 'down' | 'stable'; // Prop for trend direction
  // --- Placeholder for comparison ---
  isSelectedForCompare?: boolean;
  onCompareSelect?: (deviceId: string, selected: boolean) => void;
  // -------------
}

// --- Constants remain the same ---
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

// --- Skeleton Component (no changes) ---
function DeviceCardSkeleton() {
  // ... (skeleton code remains the same)
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


export function DeviceCard({
    device,
    loading = false,
    onClick,
    trend, // <-- NEW prop
    // --- Placeholders ---
    isSelectedForCompare,
    onCompareSelect
    // ------------------
}: DeviceCardProps) {
  const [lastTimestamp, setLastTimestamp] = useState(device?.timestamp);
  const [isPulsing, setIsPulsing] = useState(false);

  useEffect(() => {
    if (device && device.timestamp !== lastTimestamp) {
      setIsPulsing(true);
      setLastTimestamp(device.timestamp);
      const timer = setTimeout(() => setIsPulsing(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [device, device?.timestamp, lastTimestamp]);

  if (loading || !device) {
    return <DeviceCardSkeleton />;
  }

  const config = deviceTypeConfig[device.type] || deviceTypeConfig.default;
  const Icon = device.status === 'offline' ? WifiOff : config.icon;
  const statusColor = statusConfig[device.status];

  const handleCardClick = () => {
    if (onClick) {
      onClick(device.id);
    }
  };

  // --- NEW: Trend Icon Logic ---
  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus;
  const trendColor = trend === 'up' ? 'text-success' : trend === 'down' ? 'text-destructive' : 'text-muted-foreground';
  // ---------------------------

  return (
    <TooltipProvider delayDuration={100}>
      <Card
        className={cn(
          'shadow-sm hover:shadow-lg transition-all duration-300 relative', // Added relative
          onClick && 'cursor-pointer',
          isSelectedForCompare && 'ring-2 ring-primary shadow-lg' // Highlight if selected
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
              <Badge variant="outline" className="capitalize text-xs"> {/* Made text smaller */}
                {device.type.replace(/_/g, ' ')}
              </Badge>
              <Tooltip>
                <TooltipTrigger asChild>
                   <div
                    className={cn(
                      'h-3 w-3 rounded-full',
                      statusColor,
                      isPulsing && 'animate-pulse'
                    )}
                  />
                </TooltipTrigger>
                <TooltipContent>
                  <p>Status: <span className='capitalize'>{device.status}</span></p>
                  {device.timestamp && <p>Last Update: {new Date(device.timestamp).toLocaleTimeString()}</p>}
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <CardTitle className="text-lg font-semibold truncate" title={device.name}>
            {device.name}
          </CardTitle>
          <CardDescription className="text-sm truncate" title={device.id}>
            {device.id} {/* Keep ID visible but smaller */}
          </CardDescription>

          <div className="mt-4 pt-4 border-t border-border/50">
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-muted-foreground">
                {device.status === 'offline' ? 'Last seen' : 'Value'}
              </span>
              {device.status !== 'offline' ? (
                <span
                  className={cn(
                    'text-xl font-bold text-foreground',
                    isPulsing && 'animate-pulse'
                  )}
                >
                  {device.value?.toFixed(2)}{config.unit}
                </span>
              ) : (
                <span className="text-sm text-muted-foreground">
                  {new Date(device.timestamp).toLocaleString()}
                </span>
              )}
            </div>
            <div className="flex items-center justify-between mt-2"> {/* Changed to items-center */}
              <span className="text-sm text-muted-foreground flex items-center gap-1">
                 Health
                 {/* --- NEW: Trend Icon Display --- */}
                 {trend && (
                     <Tooltip>
                        <TooltipTrigger asChild>
                            <TrendIcon className={cn('h-4 w-4', trendColor)} />
                        </TooltipTrigger>
                        <TooltipContent>
                            <p>Health Trend: <span className='capitalize'>{trend}</span></p>
                        </TooltipContent>
                    </Tooltip>
                 )}
                 {/* ----------------------------- */}
              </span>
              <span
                className={cn(
                  'text-lg font-bold',
                  device.healthScore >= 80
                    ? 'text-success'
                    : device.healthScore >= 60
                    ? 'text-warning'
                    : 'text-destructive',
                  isPulsing && 'animate-pulse'
                )}
              >
                {/* Ensure healthScore is treated as a percentage */}
                {typeof device.healthScore === 'number' ? `${device.healthScore.toFixed(0)}%` : 'N/A'}
              </span>
            </div>
          </div>

          {/* --- Placeholder for Comparison --- */}
          {/* This requires parent component state management */}
          {/* <div className="absolute top-2 right-2">
            <Tooltip>
               <TooltipTrigger asChild>
                   <Button
                       variant="ghost"
                       size="icon"
                       className="h-6 w-6 text-muted-foreground hover:text-primary"
                       onClick={(e) => {
                           e.stopPropagation(); // Prevent card click
                           onCompareSelect?.(device.id, !isSelectedForCompare);
                       }}
                       aria-label={isSelectedForCompare ? 'Remove from comparison' : 'Add to comparison'}
                   >
                       {isSelectedForCompare ? <MinusCircle className="h-4 w-4 text-primary" /> : <PlusCircle className="h-4 w-4" />}
                   </Button>
               </TooltipTrigger>
               <TooltipContent>
                   <p>{isSelectedForCompare ? 'Remove from comparison' : 'Add to comparison'}</p>
               </TooltipContent>
           </Tooltip>
          </div> */}
           {/* ----------------------------- */}

        </CardContent>
      </Card>
    </TooltipProvider>
  );
}