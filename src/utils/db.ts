// src/utils/db.ts
import Dexie, { Table } from 'dexie';
import { Device, Alert, PerformanceData } from '../types/digital-twin';

// --- NEW: Interface for the KPI store ---
export interface KpiData {
  id: 'performanceData' | 'dashboardMetrics'; // Use primary key for single items
  data: any; // Store arbitrary objects
}

export class DigitalTwinDB extends Dexie {
  devices!: Table<Device>;
  alerts!: Table<Alert>;
  kpis!: Table<KpiData>; // --- NEW: Table for KPI data ---

  constructor() {
    super('digitalTwinDB');
    this.version(2).stores({
      // --- UPDATED: Schema version 2 ---
      devices: 'id, type, status, location', // Primary key 'id', index others
      alerts: 'id, deviceId, severity, timestamp, acknowledged', // Primary key 'id', index others
      kpis: 'id', // Primary key 'id'
    });
    // Removed version(1) to avoid migration errors in new setup,
    // assuming this is for demo mode.
    // If migration is needed:
    // this.version(1).stores({
    //   devices: 'id, type, status, location',
    //   alerts: 'id, deviceId, severity, timestamp, acknowledged',
    // });
    // this.version(2).stores({
    //   kpis: 'id'
    // });
  }
}

export const db = new DigitalTwinDB();