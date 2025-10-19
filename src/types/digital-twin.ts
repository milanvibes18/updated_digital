// this is my existing digital-twin.ts ocde
export interface Device {
  id: string;
  name: string;
  type: string;
  status: 'normal' | 'warning' | 'critical' | 'offline';
  location: string;
  healthScore: number;
  efficiencyScore: number;
  value: number;
  unit: string;
  timestamp: string;
}

export interface Alert {
  id: string;
  deviceId: string;
  type: string;
  severity: 'info' | 'warning' | 'critical';
  description: string;
  timestamp: string;
  acknowledged: boolean;
}

export interface PerformanceData {
  timestamp: string;
  systemHealth: number;
  efficiency: number;
  energyUsage: number;
}

export interface DashboardData {
  systemHealth: number;
  activeDevices: number;
  totalDevices: number;
  efficiency: number;
  energyUsage: number;
  energyCost: number;
  performanceData: PerformanceData[];
  statusDistribution: {
    normal: number;
    warning: number;
    critical: number;
    offline: number;
  };
}

export type TimeRange = '1h' | '4h' | '24h' | '7d' | '30d';