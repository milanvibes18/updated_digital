import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { Dashboard } from './Dashboard'
import { useDashboard } from '../hooks/useDashboard'

// Mock the custom hook
vi.mock('../hooks/useDashboard')

// Mock child components to avoid complex setup
vi.mock('./KPICard', () => ({ KPICard: (props: any) => <div data-testid="kpi-card">{props.title}</div> }))
vi.mock('./DeviceCard', () => ({ DeviceCard: () => <div data-testid="device-card"></div> }))
vi.mock('./AlertCard', () => ({ AlertCard: () => <div data-testid="alert-card"></div> }))
vi.mock('./ThresholdSettings', () => ({ ThresholdSettings: () => <div>Threshold Settings</div> }))
vi.mock('./EmailLogsView', () => ({ EmailLogsView: () => <div>Email Logs</div> }))
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  LineChart: ({ children }: any) => <div>{children}</div>,
  CartesianGrid: () => <div />,
  XAxis: () => <div />,
  YAxis: () => <div />,
  Tooltip: () => <div />,
  Line: () => <div />,
}))

// Define default mock data
const mockDashboardData = {
  systemHealth: 95,
  activeDevices: 9,
  totalDevices: 10,
  efficiency: 88,
  energyUsage: 1200,
  energyCost: 144,
  performanceData: [
    { timestamp: '00:00', systemHealth: 90, efficiency: 85 },
    { timestamp: '01:00', systemHealth: 92, efficiency: 86 },
  ],
  statusDistribution: {
    normal: 8,
    warning: 1,
    critical: 0,
    offline: 1
  }
}

const mockUseDashboard = {
  devices: Array(10).fill({}),
  alerts: [],
  dashboardData: mockDashboardData,
  timeRange: '24h',
  loading: false,
  userId: 'demo-user',
  lastUpdate: Date.now(),
  showThresholds: false,
  showEmailLogs: false,
  newCriticalAlerts: new Set(),
  isExporting: false,
  setTimeRange: vi.fn(),
  setShowThresholds: vi.fn(),
  setShowEmailLogs: vi.fn(),
  handleRefresh: vi.fn(),
  handleAcknowledgeAlert: vi.fn(),
  handleExport: vi.fn(),
  handleGenerateReport: vi.fn(),
}

describe('Dashboard Component', () => {

  beforeEach(() => {
    // Reset mocks before each test
    vi.clearAllMocks()
  })

  it('should show loading spinner when loading', () => {
    (useDashboard as vi.Mock).mockReturnValue({
      ...mockUseDashboard,
      loading: true,
    })
    
    render(<Dashboard />)
    
    expect(screen.getByText(/Loading Dashboard.../i)).toBeInTheDocument()
  })

  it('should render the dashboard title and header elements when loaded', () => {
    (useDashboard as vi.Mock).mockReturnValue(mockUseDashboard)
    
    render(<Dashboard />)
    
    expect(screen.getByText(/Digital Twin Dashboard/i)).toBeInTheDocument()
    expect(screen.getByText(/Real-time Industrial IoT Monitoring & Analytics/i)).toBeInTheDocument()
    expect(screen.getByText(/Settings/i)).toBeInTheDocument()
    expect(screen.getByText(/Report/i)).toBeInTheDocument()
  })

  it('should render KPI cards with correct data', () => {
    (useDashboard as vi.Mock).mockReturnValue(mockUseDashboard)
    
    render(<Dashboard />)
    
    const kpiCards = screen.getAllByTestId('kpi-card')
    expect(kpiCards).toHaveLength(4)
    expect(screen.getByText('System Health')).toBeInTheDocument()
    expect(screen.getByText('Active Devices')).toBeInTheDocument()
    expect(screen.getByText('Energy Usage')).toBeInTheDocument()
    expect(screen.getByText('Efficiency')).toBeInTheDocument()
  })

  it('should render the performance chart title', () => {
    (useDashboard as vi.Mock).mockReturnValue(mockUseDashboard)
    
    render(<Dashboard />)
    
    expect(screen.getByText('Performance Metrics')).toBeInTheDocument()
  })

  it('should render "No critical alerts" when there are no critical alerts', () => {
    (useDashboard as vi.Mock).mockReturnValue({
      ...mockUseDashboard,
      alerts: []
    })
    
    render(<Dashboard />)
    
    expect(screen.getByText('No critical alerts')).toBeInTheDocument()
  })
})