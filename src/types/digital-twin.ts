// src/types/digital-twin.ts

// Existing Device interface (assuming it's correct)
export interface Device {
  id: string;
  name: string;
  type: 'temperature_sensor' | 'pressure_sensor' | 'vibration_sensor' | 'humidity_sensor' | 'power_meter'; // More specific types
  status: 'normal' | 'warning' | 'critical' | 'offline';
  location: string;
  healthScore: number;
  efficiencyScore: number;
  value: number;
  unit: string;
  timestamp: string; // Consider using Date type if appropriate
  // --- NEW: Added userId as it's often needed ---
  userId?: string;
}

// Updated Alert interface
export interface Alert {
  id: string; // Ensure this is unique
  deviceId: string;
  // --- NEW: Added title and message as used in AlertCard ---
  title: string;
  message: string;
  // --- Keeping original 'type', but consider if 'title' replaces it ---
  type: string;
  severity: 'info' | 'warning' | 'critical';
  // --- Keeping original 'description', consider if 'message' replaces it ---
  description: string;
  timestamp: string; // Consider using Date type
  acknowledged: boolean;
  // --- NEW: Added userId as it's often needed ---
  userId?: string;
  // --- NEW: Optional value causing the alert ---
  value?: number;
}

// Existing PerformanceData (assuming it's correct)
export interface PerformanceData {
  timestamp: string;
  systemHealth: number;
  efficiency: number;
  energyUsage: number;
}

// Existing DashboardData (assuming structure from backend)
// NOTE: This might change based on your actual API response
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
  // --- NEW: Added fields based on Dashboard.tsx usage ---
  devices: Device[]; // Assuming the API returns devices
  alerts: Alert[];   // Assuming the API returns alerts
  timestamp: string; // Added timestamp for last update
}

// Existing TimeRange (assuming it's correct)
export type TimeRange = '1h' | '4h' | '24h' | '7d' | '30d';

// --- NEW: Added types from other components ---

// Type from ThresholdSettings.tsx
export interface AlertThreshold {
  id: string;
  deviceType: Device['type'] | 'all'; // Allow 'all' or specific device types
  metricType: string; // e.g., 'temperature', 'vibration', 'healthScore'
  operator: 'gt' | 'gte' | 'lt' | 'lte' | 'eq'; // gt = >, lte = <=, etc.
  value: number;
  severity: Alert['severity'];
  enabled: boolean | number; // Use boolean, but handle number from db if needed
  userId: string;
  createdAt?: string; // Optional timestamp
  updatedAt?: string; // Optional timestamp
}

// Type from EmailLogsView.tsx
export interface EmailAlert {
  id: string;
  alertId?: string; // Link to the original Alert
  recipient: string;
  subject: string;
  body: string;
  sentAt: string; // ISO date string
  status: 'pending' | 'sent' | 'failed';
  errorMessage?: string; // If status is 'failed'
  userId: string;
  createdAt?: string; // Optional timestamp
}

// Type from AssetManagementView.tsx (Placeholder)
export interface Asset {
  id: string;
  name: string;
  type: string;
  location: string;
  installationDate: string;
  manufacturer: string;
  model: string;
  status: string; // e.g., 'active', 'maintenance', 'decommissioned'
  lastMaintenanceDate?: string;
  nextMaintenanceDate?: string;
  userId: string;
}

// Type from AlertsHistoryView.tsx / HistoricalAnalyticsView (might reuse Alert/Device)
// Add specific types if needed for historical views