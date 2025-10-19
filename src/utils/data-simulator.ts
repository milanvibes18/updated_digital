// src/utils/data-simulator.ts
import { db } from './db'; // Import the IndexedDB instance
import { Device, Alert, PerformanceData } from '../types/digital-twin';

// (Keep all your existing helper functions like generateRandomDevice, generatePerformanceData, etc.)
// ... (omitted for brevity, assuming they exist) ...

// --- Helper Functions (assuming they exist, based on Dashboard.tsx) ---

function generateRandomValue(min: number, max: number, decimals = 2) {
  return parseFloat((Math.random() * (max - min) + min).toFixed(decimals));
}

const deviceTypes: Device['type'][] = [
  'temperature_sensor', 'pressure_sensor', 'vibration_sensor', 'humidity_sensor', 'power_meter'
];
const locations = ['Boiler Room', 'Turbine Hall', 'Pump Station', 'Control Room', 'Substation'];

function generateRandomDevice(id: string): Device {
  const type = deviceTypes[Math.floor(Math.random() * deviceTypes.length)];
  let status: Device['status'] = 'normal';
  const healthScore = generateRandomValue(40, 100, 0);
  if (healthScore < 60) status = 'critical';
  else if (healthScore < 80) status = 'warning';

  let value = 0;
  let unit = '';

  switch (type) {
    case 'temperature_sensor': value = generateRandomValue(20, 100); unit = '°C'; break;
    case 'pressure_sensor': value = generateRandomValue(100, 1000); unit = 'kPa'; break;
    case 'vibration_sensor': value = generateRandomValue(0.1, 5.0); unit = 'mm/s'; break;
    case 'humidity_sensor': value = generateRandomValue(30, 90); unit = '%'; break;
    case 'power_meter': value = generateRandomValue(100, 5000); unit = 'kW'; break;
  }

  return {
    id: `DEV-${id.padStart(3, '0')}`,
    name: `${type.replace('_', ' ').split(' ').map(w => w[0].toUpperCase() + w.slice(1)).join(' ')} ${id}`,
    type: type,
    status: Math.random() < 0.05 ? 'offline' : status,
    location: locations[Math.floor(Math.random() * locations.length)],
    healthScore: healthScore,
    efficiencyScore: generateRandomValue(70, 95),
    value: value,
    unit: unit,
    timestamp: new Date().toISOString(),
  };
}

function generatePerformanceData(numPoints: number): PerformanceData[] {
  const data: PerformanceData[] = [];
  let lastHealth = generateRandomValue(80, 95);
  let lastEfficiency = generateRandomValue(85, 98);
  const now = new Date();

  for (let i = numPoints - 1; i >= 0; i--) {
    const timestamp = new Date(now.getTime() - i * 60000 * 5); // 5 min intervals
    
    lastHealth += generateRandomValue(-1.5, 1.5);
    lastEfficiency += generateRandomValue(-1, 1);
    lastHealth = Math.max(50, Math.min(100, lastHealth));
    lastEfficiency = Math.max(60, Math.min(100, lastEfficiency));

    data.push({
      timestamp: timestamp.toISOString(),
      systemHealth: parseFloat(lastHealth.toFixed(1)),
      efficiency: parseFloat(lastEfficiency.toFixed(1)),
      energyUsage: generateRandomValue(1500, 4500),
    });
  }
  return data;
}

// --- End of assumed helper functions ---


const SIMULATION_INTERVAL = 5000; // 5 seconds
let simulationIntervalId: number | null = null;

let simulatedDevices: Device[] = Array.from({ length: 12 }, (_, i) => generateRandomDevice((i + 1).toString()));
let simulatedAlerts: Alert[] = [];
let performanceData: PerformanceData[] = generatePerformanceData(100);

// --- UPDATED: Main Simulation Logic ---
async function runSimulationTick() {
  const now = new Date();
  
  // 1. Update Devices
  simulatedDevices = simulatedDevices.map(device => {
    // Small chance of status change
    if (Math.random() < 0.1) {
      if (device.status === 'offline' && Math.random() < 0.5) {
        device.status = 'normal'; // Came back online
        device.healthScore = generateRandomValue(80, 95);
      } else if (device.status !== 'offline' && Math.random() < 0.1) {
        device.status = 'offline'; // Went offline
      }
    }

    if (device.status !== 'offline') {
      // Fluctuate values and health
      device.value = generateRandomValue(device.value * 0.98, device.value * 1.02);
      device.healthScore = Math.max(40, Math.min(100, device.healthScore + generateRandomValue(-0.5, 0.5)));
      
      // Update status based on health
      if (device.healthScore < 60) device.status = 'critical';
      else if (device.healthScore < 80) device.status = 'warning';
      else device.status = 'normal';

      // Check for new alerts
      if (device.status === 'critical' && Math.random() < 0.3) {
        const newAlert: Alert = {
          id: `ALERT-${Date.now()}-${device.id}`,
          deviceId: device.id,
          title: `${device.name} Critical Health`,
          message: `Device health dropped to ${device.healthScore.toFixed(0)}%. Immediate attention required.`,
          type: 'DeviceHealth', // Using type as category
          severity: 'critical',
          description: `Current value: ${device.value.toFixed(2)}${device.unit}`, // Using description for details
          timestamp: now.toISOString(),
          acknowledged: false,
        };
        simulatedAlerts = [newAlert, ...simulatedAlerts].slice(0, 50); // Add to top, limit 50
      }
    }
    device.timestamp = now.toISOString();
    return device;
  });

  // 2. Update Performance Data
  const lastPerf = performanceData[performanceData.length - 1];
  const avgHealth = simulatedDevices.reduce((acc, d) => acc + d.healthScore, 0) / simulatedDevices.length;
  const avgEfficiency = simulatedDevices.reduce((acc, d) => acc + d.efficiencyScore, 0) / simulatedDevices.length;

  performanceData.push({
    timestamp: now.toISOString(),
    systemHealth: parseFloat(avgHealth.toFixed(1)),
    efficiency: parseFloat(avgEfficiency.toFixed(1)),
    energyUsage: lastPerf.energyUsage + generateRandomValue(-100, 100),
  });
  performanceData.shift(); // Keep array size constant

  try {
    // --- NEW: Write data to IndexedDB ---
    await db.devices.bulkPut(simulatedDevices);
    await db.alerts.bulkPut(simulatedAlerts);
    
    // Store performance data (assuming db.ts is updated to handle it)
    // For simplicity, we'll store it in a single 'kpi' entry
    await db.kpis.put({
      id: 'performanceData',
      data: performanceData
    });
    await db.kpis.put({
      id: 'dashboardMetrics',
      data: {
        systemHealth: avgHealth,
        activeDevices: simulatedDevices.filter(d => d.status !== 'offline').length,
        totalDevices: simulatedDevices.length,
        efficiency: avgEfficiency,
        energyUsage: performanceData[performanceData.length - 1].energyUsage,
        energyCost: performanceData[performanceData.length - 1].energyUsage * 0.12, // Assuming $0.12/kWh
        statusDistribution: {
          normal: simulatedDevices.filter(d => d.status === 'normal').length,
          warning: simulatedDevices.filter(d => d.status === 'warning').length,
          critical: simulatedDevices.filter(d => d.status === 'critical').length,
          offline: simulatedDevices.filter(d => d.status === 'offline').length,
        },
      }
    });

  } catch (error) {
    console.error('Demo Mode: Error writing to IndexedDB:', error);
  }

  // --- Emit events for any live listeners (like Dashboard) ---
  // This is optional if dashboard now polls from api.ts (which reads from db.ts)
  // but good to keep for immediate UI updates if we add listeners.
  document.dispatchEvent(new CustomEvent('demoDataUpdated'));
}

// --- UPDATED: Exportable start/stop functions ---
export async function startSimulation() {
  if (simulationIntervalId) {
    console.log('Simulation already running.');
    return;
  }
  
  console.log('Starting demo mode simulation...');
  // Run first tick immediately to populate DB
  await runSimulationTick(); 
  // Then start interval
  simulationIntervalId = window.setInterval(runSimulationTick, SIMULATION_INTERVAL);
}

export function stopSimulation() {
  if (simulationIntervalId) {
    console.log('Stopping demo mode simulation.');
    clearInterval(simulationIntervalId);
    simulationIntervalId = null;
  }
}