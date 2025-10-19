// src/utils/api.ts (RECOMMENDED UPDATE)
import { db } from './db';
import { DashboardData, Device, Alert, AlertThreshold, EmailAlert } from '../types/digital-twin';

// Use environment variable for API base URL, fallback for development
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000/api';

// --- UPDATED: fetchWithAuth ---
// This function now relies on HttpOnly cookies
async function fetchWithAuth(url: string, options: RequestInit = {}) {
  // --- REMOVED: Token logic ---
  // const token = localStorage.getItem('jwt_token');

  const defaultHeaders: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  // --- REMOVED: Authorization header ---
  // if (token) {
  //   defaultHeaders['Authorization'] = `Bearer ${token}`;
  // }

  const response = await fetch(`${API_BASE_URL}${url}`, {
    ...options,
    headers: defaultHeaders,
    // --- NEW: Send credentials (cookies) with every request ---
    credentials: 'include', 
  });

  if (!response.ok) {
    let errorData;
    try {
      errorData = await response.json();
    } catch (e) {
      errorData = { error: `HTTP error! Status: ${response.status}` };
    }
    // Use the error message from the backend JSON response if available
    throw new Error(errorData.error || errorData.message || `HTTP error! Status: ${response.status}`);
  }

  // Handle '204 No Content' responses
  if (response.status === 204) {
    return null;
  }
  
  return response.json();
}

// --- UPDATED: API functions now check for Demo Mode ---

export async function loginUser(credentials: { [key: string]: string }) {
  // Login is the only function that *doesn't* check for demo mode
  // and *doesn't* use fetchWithAuth (it establishes the session)
  const response = await fetch(`${API_BASE_URL}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Invalid username or password');
  }
  // We no longer return the token, just the success data
  return response.json(); 
}

// --- NEW: Function to get user identity from session ---
export async function getCurrentUser() {
  // This endpoint should return the user's identity based on their session cookie
  return fetchWithAuth('/auth/me'); 
}


export async function getDashboardData(): Promise<DashboardData> {
  // --- NEW: Demo Mode Check ---
  if (localStorage.getItem('demo_mode') === 'true') {
    console.log('API: Running in Demo Mode (getDashboardData)');
    const [devices, performance, metrics] = await Promise.all([
      db.devices.toArray(),
      db.kpis.get('performanceData'),
      db.kpis.get('dashboardMetrics')
    ]);
    
    // Combine data into DashboardData format
    const dashboardMetrics = metrics?.data || {};
    return {
      ...dashboardMetrics,
      performanceData: performance?.data || [],
      devices: devices || [],
      alerts: [], // Alerts handled by getAlerts
      timestamp: new Date().toISOString(),
    } as DashboardData;
  }
  // --- End Demo Mode Check ---

  return fetchWithAuth('/dashboard');
}

export async function getAlerts(userId: string, limit: number): Promise<Alert[]> {
  // --- NEW: Demo Mode Check ---
  if (localStorage.getItem('demo_mode') === 'true') {
    console.log('API: Running in Demo Mode (getAlerts)');
    return db.alerts.orderBy('timestamp').reverse().limit(limit).toArray();
  }
  // --- End Demo Mode Check ---
  
  // Assuming API uses query params
  return fetchWithAuth(`/alerts?userId=${userId}&limit=${limit}`);
}

export async function acknowledgeAlert(alertId: string): Promise<Alert> {
  // --- NEW: Demo Mode Check ---
  if (localStorage.getItem('demo_mode') === 'true') {
    console.log('API: Running in Demo Mode (acknowledgeAlert)');
    await db.alerts.update(alertId, { acknowledged: true });
    const updatedAlert = await db.alerts.get(alertId);
    if (!updatedAlert) throw new Error("Alert not found in demo DB");
    return updatedAlert;
  }
  // --- End Demo Mode Check ---

  return fetchWithAuth(`/alerts/acknowledge/${alertId}`, { method: 'POST' });
}

export async function getDevice(deviceId: string): Promise<Device> {
  // --- NEW: Demo Mode Check ---
  if (localStorage.getItem('demo_mode') === 'true') {
     console.log('API: Running in Demo Mode (getDevice)');
     const device = await db.devices.get(deviceId);
     if (!device) throw new Error("Device not found in demo DB");
     return device;
  }
  // --- End Demo Mode Check ---

  return fetchWithAuth(`/devices/${deviceId}`);
}

// ... (Keep other functions like generateReport, exportData, etc.) ...
// ... (They will now use fetchWithAuth and send cookies) ...

export async function generateReport(): Promise<{ report_path: string; filename: string }> {
  return fetchWithAuth('/reports/generate', { method: 'POST' });
}

export async function exportData(format: 'csv' | 'json', days: number): Promise<{ export_path: string; filename: string }> {
  return fetchWithAuth(`/export?format=${format}&days=${days}`, { method: 'GET' });
}

// --- THRESHOLD and EMAIL LOG functions ---
// (These also need demo mode counterparts)

export async function getAlertThresholds(userId: string): Promise<AlertThreshold[]> {
  // --- NEW: Demo Mode Check ---
  if (localStorage.getItem('demo_mode') === 'true') {
    console.log('API: Running in Demo Mode (getAlertThresholds)');
    // Demo mode thresholds (add to db.ts if you want to store them)
    return [
      { id: '1', deviceType: 'temperature_sensor', metricType: 'value', operator: 'gt', value: 80, severity: 'warning', enabled: 1, userId: 'demo_user' },
      { id: '2', deviceType: 'all', metricType: 'healthScore', operator: 'lt', value: 60, severity: 'critical', enabled: 1, userId: 'demo_user' },
    ];
  }
  // --- End Demo Mode Check ---
  return fetchWithAuth(`/settings/thresholds?userId=${userId}`);
}

export async function saveAlertThreshold(threshold: Omit<AlertThreshold, 'id'>): Promise<AlertThreshold> {
  // --- NEW: Demo Mode Check ---
  if (localStorage.getItem('demo_mode') === 'true') {
    console.log('API: Running in Demo Mode (saveAlertThreshold)');
    const newThreshold = { ...threshold, id: `DEMO-${Date.now()}` };
    // (add to db.kpis if you want to store)
    return newThreshold;
  }
  // --- End Demo Mode Check ---
  return fetchWithAuth('/settings/thresholds', {
    method: 'POST',
    body: JSON.stringify(threshold),
  });
}

export async function updateAlertThreshold(threshold: AlertThreshold): Promise<AlertThreshold> {
  // --- NEW: Demo Mode Check ---
  if (localStorage.getItem('demo_mode') === 'true') {
    console.log('API: Running in Demo Mode (updateAlertThreshold)');
    // (update in db.kpis if stored)
    return threshold;
  }
  // --- End Demo Mode Check ---
  return fetchWithAuth(`/settings/thresholds/${threshold.id}`, {
    method: 'PUT',
    body: JSON.stringify(threshold),
  });
}

export async function deleteAlertThreshold(thresholdId: string): Promise<{ message: string }> {
  // --- NEW: Demo Mode Check ---
  if (localStorage.getItem('demo_mode') === 'true') {
    console.log('API: Running in Demo Mode (deleteAlertThreshold)');
    // (delete in db.kpis if stored)
    return { message: 'Demo threshold deleted' };
  }
  // --- End Demo Mode Check ---
  return fetchWithAuth(`/settings/thresholds/${thresholdId}`, {
    method: 'DELETE',
  });
}

export async function getEmailLogs(userId: string, limit: number): Promise<EmailAlert[]> {
  // --- NEW: Demo Mode Check ---
  if (localStorage.getItem('demo_mode') === 'true') {
     console.log('API: Running in Demo Mode (getEmailLogs)');
     return [
       { id: '1', recipient: 'admin@demo.com', subject: 'Critical Alert: DEV-003', sentAt: new Date().toISOString(), status: 'sent', userId: 'demo_user', body: 'Demo alert email body.' },
       { id: '2', recipient: 'admin@demo.com', subject: 'Warning Alert: DEV-001', sentAt: new Date(Date.now() - 3600000).toISOString(), status: 'failed', errorMessage: 'Demo SMTP error', userId: 'demo_user', body: 'Demo alert email body.' },
     ];
  }
  // --- End Demo Mode Check ---
  return fetchWithAuth(`/logs/email?userId=${userId}&limit=${limit}`);
}