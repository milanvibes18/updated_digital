// Dashboard.test.tsx
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { BrowserRouter } from 'react-router-dom';
import { Dashboard } from '../components/Dashboard'; // ✅ Adjust path if needed
import { useDashboard } from '../hooks/useDashboard';
import {
  sampleAlerts,
  sampleDevices,
  sampleDashboardData,
} from './_mocks/sample-data';

// --- ✅ Mock the useDashboard hook ---
vi.mock('../hooks/useDashboard', () => ({
  useDashboard: vi.fn(),
}));

// --- ✅ Mock ResizeObserver (used by charting libs like Recharts) ---
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// --- ✅ Wrapper component for Router context ---
const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <BrowserRouter>{children}</BrowserRouter>
);

describe('Dashboard Component', () => {
  // --- ✅ Automatic cleanup after each test ---
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  // Helper alias for cleaner mock typing
  const mockedUseDashboard = useDashboard as unknown as ReturnType<typeof vi.fn>;

  // ---------------------------------------------------------------------------
  // 🧩 Test 1: Snapshot + Layout verification
  // ---------------------------------------------------------------------------
  it('renders initial layout correctly and matches snapshot', async () => {
    // Arrange
    mockedUseDashboard.mockReturnValue({
      devices: sampleDevices,
      alerts: sampleAlerts,
      dashboardData: sampleDashboardData,
      loading: false,
      timeRange: '1h',
      userId: 'test-user',
      lastUpdate: new Date().toISOString(),
      newCriticalAlerts: new Set(),
      handleRefresh: vi.fn(),
      handleAcknowledgeAlert: vi.fn(),
    });

    // Act
    const { container } = render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    // Assert
    await waitFor(() => {
      expect(
        screen.getByRole('heading', { name: /digital twin dashboard/i })
      ).toBeVisible();
    });

    // Snapshot test for layout regression (keep it minimal)
    expect(container.firstChild).toMatchSnapshot();
  });

  // ---------------------------------------------------------------------------
  // 🧩 Test 2: Real-time critical alert update
  // ---------------------------------------------------------------------------
  it('displays new critical alerts when data updates', async () => {
    // Arrange (initial state — no critical alerts)
    const initialAlerts = sampleAlerts.filter((a) => a.severity !== 'critical');

    mockedUseDashboard.mockReturnValue({
      devices: sampleDevices,
      alerts: initialAlerts,
      dashboardData: sampleDashboardData,
      loading: false,
      timeRange: '1h',
      newCriticalAlerts: new Set(),
      handleAcknowledgeAlert: vi.fn(),
    });

    // Act: Render component
    const { rerender } = render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    // Assert: No critical alert visible initially
    expect(
      screen.queryByText(/exceeds critical threshold/i)
    ).not.toBeInTheDocument();

    // Arrange (update with new critical alert)
    const newCriticalAlert = {
      id: 'alert-crit-1',
      deviceId: 'device-1',
      deviceName: 'Temperature Sensor #1',
      severity: 'critical',
      message: 'Temperature exceeds critical threshold: 98.50 °C.',
      timestamp: new Date().toISOString(),
      acknowledged: false,
    };

    mockedUseDashboard.mockReturnValue({
      devices: sampleDevices,
      alerts: [...initialAlerts, newCriticalAlert],
      dashboardData: sampleDashboardData,
      loading: false,
      timeRange: '1h',
      newCriticalAlerts: new Set(['alert-crit-1']),
      handleAcknowledgeAlert: vi.fn(),
    });

    // Act: Rerender with updated data
    rerender(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    // Assert: Wait for the new critical alert to appear
    await waitFor(() => {
      expect(
        screen.getByText(/temperature exceeds critical threshold/i)
      ).toBeInTheDocument();
    });

    // Optional: Verify critical alert count badge updated
    expect(screen.getByText('1')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // 🧩 Test 3: Loading state rendering
  // ---------------------------------------------------------------------------
  it('shows loading indicator when dashboard data is loading', () => {
    // Arrange
    mockedUseDashboard.mockReturnValue({
      devices: [],
      alerts: [],
      dashboardData: null,
      loading: true,
      timeRange: '1h',
      newCriticalAlerts: new Set(),
      handleAcknowledgeAlert: vi.fn(),
    });

    // Act
    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    // Assert
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // 🧩 Test 4: Alert acknowledgment handler
  // ---------------------------------------------------------------------------
  it('calls handleAcknowledgeAlert when alert acknowledgment button is clicked', async () => {
    const mockHandleAcknowledge = vi.fn();

    const alert = {
      id: 'alert-crit-2',
      deviceId: 'device-2',
      deviceName: 'Pressure Sensor #2',
      severity: 'critical',
      message: 'Pressure exceeds critical threshold.',
      timestamp: new Date().toISOString(),
      acknowledged: false,
    };

    mockedUseDashboard.mockReturnValue({
      devices: sampleDevices,
      alerts: [alert],
      dashboardData: sampleDashboardData,
      loading: false,
      timeRange: '1h',
      newCriticalAlerts: new Set(['alert-crit-2']),
      handleAcknowledgeAlert: mockHandleAcknowledge,
    });

    render(
      <TestWrapper>
        <Dashboard />
      </TestWrapper>
    );

    // Find and click the acknowledge button
    const acknowledgeButton = await screen.findByRole('button', {
      name: /acknowledge/i,
    });
    acknowledgeButton.click();

    // Assert callback fired with correct alert ID
    expect(mockHandleAcknowledge).toHaveBeenCalledWith('alert-crit-2');
  });
});
