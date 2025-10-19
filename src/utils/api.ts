/**
 * API Client for interacting with the Flask Digital Twin backend.
 */

// Use environment variable for the base API URL (e.g., from .env file via Vite)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000/api';
// Public base URL (for non-API endpoints like /health)
const PUBLIC_BASE_URL = API_BASE_URL.replace('/api', ''); // Assumes /api suffix

/**
 * Custom Error class for API fetch errors.
 * Includes the HTTP status code for more specific error handling.
 */
export class ApiError extends Error {
    status: number;
    details?: any; // Optional field for extra error details from response body

    constructor(message: string, status: number, details?: any) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.details = details;
    }
}

/**
 * Retrieves the JWT token from localStorage.
 */
const getToken = (): string | null => {
    return localStorage.getItem('jwt_token');
};

/**
 * Performs an authenticated fetch request to the API.
 * Handles token attachment, 401 errors (redirects to login), and includes AbortSignal support.
 * Throws ApiError on failure.
 */
const apiFetch = async (
    endpoint: string,
    options: RequestInit = {},
    usePublicUrl: boolean = false // Flag to use the public base URL
): Promise<any> => {
    const token = getToken();
    const url = usePublicUrl ? `${PUBLIC_BASE_URL}${endpoint}` : `${API_BASE_URL}${endpoint}`;

    const headers = new Headers(options.headers || {});
    // Set default Content-Type if not already set and body exists
    if (options.body && !headers.has('Content-Type')) {
        headers.set('Content-Type', 'application/json');
    }

    // Add Authorization header if token exists and not hitting a public URL
    if (token && !usePublicUrl) {
        headers.set('Authorization', `Bearer ${token}`);
    }

    try {
        const response = await fetch(url, {
            ...options,
            headers,
            signal: options.signal, // Pass AbortSignal if provided
        });

        if (response.status === 401 && !usePublicUrl) {
            // Unauthorized or token expired (only redirect if it was a protected endpoint)
            localStorage.removeItem('jwt_token');
            // Using window.location is simple and reliable.
            // For smoother SPA transitions, consider using your router's navigation method
            // (e.g., navigate('/login')) in the component catching this error.
            window.location.href = '/login'; // Force redirect to login
            throw new ApiError('Unauthorized', response.status);
        }

        if (!response.ok) {
            let errorData: any = { message: `HTTP error! Status: ${response.status}` };
            try {
                 // Try to parse JSON error details from the backend
                errorData = await response.json();
            } catch (e) {
                 // If response is not JSON or empty, use status text or default message
                 errorData = { message: response.statusText || `HTTP error! Status: ${response.status}` };
            }
             // Use 'error' or 'message' field from backend response if available
            const errorMessage = errorData.error || errorData.message || `HTTP error! Status: ${response.status}`;
            throw new ApiError(errorMessage, response.status, errorData);
        }

         // Handle cases with no JSON body (e.g., 204 No Content)
         if (response.status === 204 || response.headers.get('content-length') === '0') {
             return null; // Or return an empty object/true based on API design
         }

        // Attempt to parse JSON only if there's content
        return await response.json();

    } catch (error) {
        // Log network errors or AbortError differently if needed
        if (error instanceof ApiError) {
             // Re-throw ApiError to be handled by calling code
             throw error;
        } else if (error instanceof DOMException && error.name === 'AbortError') {
             console.log('API Fetch Aborted:', endpoint);
             throw error; // Re-throw AbortError
        } else {
             // Handle generic network errors (e.g., DNS, CORS, offline)
             console.error('API Fetch Network/Generic Error:', error);
             throw new Error('Network error or request failed. Please check your connection.');
        }
    }
};

// --- API Methods ---

/**
 * Logs in a user.
 * @param username
 * @param password
 * @param signal Optional AbortSignal
 * @returns { access_token: string }
 */
export const loginUser = async (username: string, password: string, signal?: AbortSignal) => {
    const data = await apiFetch('/login', {
        method: 'POST',
        body: JSON.stringify({ username, password }),
        signal,
    });

    if (data.access_token) {
        localStorage.setItem('jwt_token', data.access_token);
    }
    return data;
};

/**
 * Registers a new user.
 * @param username
 * @param password
 * @param email
 * @param signal Optional AbortSignal
 */
export const registerUser = async (username: string, password: string, email: string, signal?: AbortSignal) => {
    return await apiFetch('/register', {
        method: 'POST',
        body: JSON.stringify({ username, password, email }),
        signal,
    });
};

/**
 * Logs out the user (clears token locally, informs backend).
 * @param signal Optional AbortSignal
 */
export const logoutUser = async (signal?: AbortSignal) => {
     try {
         // Inform the backend (optional, depends on backend session management)
         await apiFetch('/logout', { method: 'POST', signal });
     } catch (error) {
         // Log error but proceed with local logout even if backend call fails
         console.error("Backend logout call failed:", error);
     } finally {
         // Always remove local token
         localStorage.removeItem('jwt_token');
     }
};

/**
 * Fetches the current user's identity.
 * @param signal Optional AbortSignal
 */
export const getCurrentUser = async (signal?: AbortSignal) => {
     return await apiFetch('/whoami', { signal });
};


/**
 * Fetches the main dashboard data.
 * @param signal Optional AbortSignal
 */
export const getDashboardData = async (signal?: AbortSignal) => {
    return await apiFetch('/dashboard_data', { signal });
};

/**
 * Fetches all device data.
 * @param signal Optional AbortSignal
 */
export const getDevices = async (signal?: AbortSignal) => {
    return await apiFetch('/devices', { signal });
};

/**
 * Fetches data for a specific device.
 * @param deviceId The ID of the device.
 * @param signal Optional AbortSignal
 */
export const getDeviceDetails = async (deviceId: string, signal?: AbortSignal) => {
     return await apiFetch(`/device/${deviceId}`, { signal });
};

/**
 * Fetches the public system status (health check).
 * @param signal Optional AbortSignal
 */
export const getSystemStatus = async (signal?: AbortSignal) => {
    // Calls the public /health endpoint
    return await apiFetch('/health', { signal }, true); // Use public URL
};

/**
 * Fetches system health scores (protected endpoint).
 * @param signal Optional AbortSignal
 */
export const getHealthScores = async (signal?: AbortSignal) => {
    // Calls the protected /api/health_scores endpoint
    return await apiFetch('/health_scores', { signal });
};


/**
 * Fetches current alerts. Can filter by severity.
 * @param limit Max number of alerts to fetch.
 * @param severity Optional severity filter ('info', 'warning', 'critical').
 * @param signal Optional AbortSignal
 */
export const getAlerts = async (limit: number = 10, severity?: string, signal?: AbortSignal) => {
     let endpoint = `/alerts?limit=${limit}`;
     if (severity) {
         endpoint += `&severity=${severity}`;
     }
    return await apiFetch(endpoint, { signal });
};

/**
 * Fetches AI-generated recommendations.
 * @param signal Optional AbortSignal
 */
export const getRecommendations = async (signal?: AbortSignal) => {
    return await apiFetch('/recommendations', { signal });
};

/**
 * Fetches predictions for a specific device.
 * @param deviceId
 * @param signal Optional AbortSignal
 */
export const getPredictions = async (deviceId: string, signal?: AbortSignal) => {
    return await apiFetch(`/predictions?device_id=${deviceId}`, { signal });
};

/**
 * Fetches historical data for a specific device.
 * @param deviceId
 * @param hours Number of past hours to fetch.
 * @param metric The specific metric (e.g., 'value', 'temperature').
 * @param signal Optional AbortSignal
 */
export const getHistoricalData = async (deviceId: string, hours: number = 24, metric: string = 'value', signal?: AbortSignal) => {
     return await apiFetch(`/historical_data?device_id=${deviceId}&hours=${hours}&metric=${metric}`, { signal });
};

/**
 * Fetches system metrics (CPU, memory etc.).
 * @param signal Optional AbortSignal
 */
export const getSystemMetrics = async (signal?: AbortSignal) => {
     return await apiFetch('/system_metrics', { signal });
};


/**
 * Triggers report generation on the backend.
 * @param signal Optional AbortSignal
 * @returns { success: boolean, report_path: string, message: string }
 */
export const generateReport = async (signal?: AbortSignal) => {
    return await apiFetch('/generate_report', { signal });
};

/**
 * Triggers data export on the backend.
 * @param format 'json' or 'csv'.
 * @param days Number of days to export.
 * @param signal Optional AbortSignal
 * @returns { success: boolean, export_path: string, filename: string, message: string }
 */
export const exportData = async (format: 'json' | 'csv' = 'json', days: number = 7, signal?: AbortSignal) => {
    return await apiFetch(`/export_data?format=${format}&days=${days}`, { signal });
};

// You might add more specific API calls here as your backend evolves
// e.g., updateDeviceConfig, acknowledgeAlert, etc.