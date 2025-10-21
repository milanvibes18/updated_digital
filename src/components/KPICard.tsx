// src/components/KPICard.tsx
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Skeleton } from './ui/skeleton';
import { LucideIcon, TrendingUp, TrendingDown, Minus } from 'lucide-react'; // Added trend icons
import { cn } from '../utils/cn';
import React from 'react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip'; // Added Tooltip

interface KPICardProps {
  title: string;
  value: string;
  subtitle?: string;
  icon: LucideIcon;
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'; // Added info variant
  loading?: boolean;
  // --- NEW: Added trend and lastUpdate props ---
  trend?: {
    value: string;
    direction: 'up' | 'down' | 'stable';
  };
  lastUpdate?: string | number | Date; // Accept different date formats
  // ------------------------------------------
}

const variantColors = {
  default: 'text-foreground',
  success: 'text-success', // Use theme colors
  warning: 'text-warning',
  danger: 'text-destructive',
  info: 'text-primary' // Use primary for info
};

// --- NEW: Trend Icon Mapping ---
const trendIcons = {
    up: TrendingUp,
    down: TrendingDown,
    stable: Minus,
};
const trendColors = {
    up: 'text-success',
    down: 'text-destructive',
    stable: 'text-muted-foreground',
};
// ----------------------------

export function KPICard({
  title,
  value,
  subtitle,
  icon: Icon,
  variant = 'default',
  loading = false,
  trend,         // <-- NEW prop
  lastUpdate,    // <-- NEW prop
}: KPICardProps) {
  const colorClass = variantColors[variant] || variantColors.default;

  // --- NEW: Format lastUpdate ---
  const formattedLastUpdate = lastUpdate
    ? new Date(lastUpdate).toLocaleTimeString()
    : null;
  // ----------------------------

  // --- NEW: Get Trend Icon and Color ---
  const TrendIcon = trend ? trendIcons[trend.direction] : null;
  const trendColor = trend ? trendColors[trend.direction] : '';
  // ------------------------------------

  return (
    <TooltipProvider delayDuration={100}>
      <Card className="shadow-sm hover:shadow-md transition-shadow relative overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">{title}</CardTitle>
          <Icon
            className={cn(
              'h-4 w-4 text-muted-foreground',
              !loading && colorClass
            )}
          />
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-8 w-3/4" />
              {subtitle && <Skeleton className="h-4 w-1/2" />}
              {/* Add skeleton for trend/update */}
              <Skeleton className="h-3 w-1/3 mt-1" />
            </div>
          ) : (
            <React.Fragment>
              <div className={cn('text-2xl font-bold', colorClass)}>{value}</div>
              {subtitle && (
                <p className="text-xs text-muted-foreground">{subtitle}</p>
              )}
              {/* --- NEW: Display Trend and Last Update --- */}
              <div className="flex items-center text-xs text-muted-foreground mt-1 space-x-2">
                {trend && TrendIcon && (
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <div className={cn("flex items-center", trendColor)}>
                                <TrendIcon className="h-3 w-3 mr-0.5" />
                                <span>{trend.value}</span>
                            </div>
                         </TooltipTrigger>
                         <TooltipContent>
                           <p>Trend vs previous period</p>
                         </TooltipContent>
                    </Tooltip>
                )}
                {formattedLastUpdate && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                       <span className='opacity-70'>Updated: {formattedLastUpdate}</span>
                    </TooltipTrigger>
                     <TooltipContent>
                       <p>Last data received at {new Date(lastUpdate!).toLocaleString()}</p>
                     </TooltipContent>
                   </Tooltip>
                )}
              </div>
              {/* ------------------------------------------ */}
            </React.Fragment>
          )}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
}