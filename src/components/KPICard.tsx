import { Card } from './ui/card'
import { cn } from '../utils/cn'
import React from 'react'

// LucideIcon type - generic icon component type
type LucideIcon = React.ComponentType<{
  size?: string | number
  color?: string
  strokeWidth?: string | number
  className?: string
}>

interface KPICardProps {
  title: string
  value: string | number
  subtitle?: string
  trend?: {
    value: string
    direction: 'up' | 'down' | 'neutral'
  }
  icon: LucideIcon
  className?: string
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'
  loading?: boolean
}

const variantStyles = {
  default: 'border-border',
  success: 'border-green-500/20 bg-green-500/5',
  warning: 'border-yellow-500/20 bg-yellow-500/5',
  danger: 'border-red-500/20 bg-red-500/5',
  info: 'border-blue-500/20 bg-blue-500/5'
}

const iconStyles = {
  default: 'text-muted-foreground',
  success: 'text-green-500',
  warning: 'text-yellow-500',
  danger: 'text-red-500',
  info: 'text-blue-500'
}

export function KPICard({
  title,
  value,
  subtitle,
  trend,
  icon: Icon,
  className,
  variant = 'default',
  loading = false
}: KPICardProps) {
  return (
    <Card className={cn(
      'p-6 transition-all duration-300 hover:shadow-lg',
      variantStyles[variant],
      className
    )}>
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <p className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            {title}
          </p>
          {loading ? (
            <div className="h-8 w-20 bg-muted animate-pulse rounded" />
          ) : (
            <p className="text-3xl font-bold text-foreground">
              {value}
            </p>
          )}
          {trend && (
            <div className={cn(
              "flex items-center text-sm font-medium",
              trend.direction === 'up' && "text-green-500",
              trend.direction === 'down' && "text-red-500",
              trend.direction === 'neutral' && "text-muted-foreground"
            )}>
              <span className="mr-1">
                {trend.direction === 'up' && '↗'}
                {trend.direction === 'down' && '↘'}
                {trend.direction === 'neutral' && '→'}
              </span>
              {trend.value}
            </div>
          )}
          {subtitle && (
            <p className="text-sm text-muted-foreground">
              {subtitle}
            </p>
          )}
        </div>
        <Icon className={cn('h-8 w-8', iconStyles[variant])} />
      </div>
    </Card>
  )
}