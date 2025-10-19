// src/components/KPICard.tsx
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Skeleton } from './ui/skeleton'; // Import Skeleton
import { LucideIcon } from 'lucide-react';
import { cn } from '../utils/cn';
import React from 'react'; // Import React for fragments

interface KPICardProps {
  title: string;
  value: string;
  subtitle?: string;
  icon: LucideIcon;
  variant?: 'default' | 'success' | 'warning' | 'danger';
  // --- NEW: Added loading prop ---
  loading?: boolean;
}

const variantColors = {
  default: 'text-foreground',
  success: 'text-success',
  warning: 'text-warning',
  danger: 'text-destructive',
};

export function KPICard({
  title,
  value,
  subtitle,
  icon: Icon,
  variant = 'default',
  loading = false, // Default to false
}: KPICardProps) {
  const colorClass = variantColors[variant];

  return (
    <Card className="shadow-sm hover:shadow-md transition-shadow relative overflow-hidden">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon
          className={cn(
            'h-4 w-4 text-muted-foreground',
            !loading && colorClass // Only apply color if not loading
          )}
        />
      </CardHeader>
      <CardContent>
        {loading ? (
          // --- NEW: Loading Skeleton State ---
          <div className="space-y-2">
            <Skeleton className="h-8 w-3/4" />
            {subtitle && <Skeleton className="h-4 w-1/2" />}
          </div>
        ) : (
          // --- Existing Content ---
          <React.Fragment>
            <div className={cn('text-2xl font-bold', colorClass)}>{value}</div>
            {subtitle && (
              <p className="text-xs text-muted-foreground">{subtitle}</p>
            )}
          </React.Fragment>
        )}
      </CardContent>
    </Card>
  );
}