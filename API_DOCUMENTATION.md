# API Documentation v2

This document details the REST API (Flask) and WebSocket API (Flask-SocketIO) used for communication between the frontend and backend of the Digital Twin System.

**Authentication:** Most REST endpoints require a valid **JSON Web Token (JWT)**. The token can be sent via an `Authorization: Bearer <token>` header or via secure, HttpOnly cookies managed by the backend after login. WebSocket connections also require authentication during the handshake.

**Base URL:** `/api`

---

## 1. REST API Endpoints

### Authentication (`/api/auth`)

#### `POST /api/auth/register`

* **Description:** Registers a new user.
* **Request Body (JSON):**
    ```json
    {
      "username": "newuser",
      "password": "strongpassword123",
      "email": "user@example.com"
    }
    ```
* **Success Response (201 Created):**
    ```json
    {
      "message": "User registered successfully"
    }
    ```
* **Error Responses:**
    * `400 Bad Request`: Validation errors (e.g., password too short, invalid email).
    * `409 Conflict`: Username or email already exists.
* **Authentication:** None required.
* **Rate Limit:** 5 per hour.

#### `POST /api/auth/login`

* **Description:** Authenticates a user and returns JWTs (access token in body, access + refresh tokens set as HttpOnly cookies).
* **Request Body (JSON):**
    ```json
    {
      "username": "testuser",
      "password": "password123"
    }
    ```
* **Success Response (200 OK):**
    ```json
    {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...", // For client-side storage if needed
      "user": {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "created_at": "...",
        "role": "user"
      }
    }
    ```
    *(Backend also sets `access_token_cookie` and `refresh_token_cookie`)*
* **Error Responses:**
    * `400 Bad Request`: Validation errors.
    * `401 Unauthorized`: Invalid credentials.
* **Authentication:** None required.
* **Rate Limit:** 10 per minute.

#### `POST /api/auth/refresh`

* **Description:** Refreshes an expired access token using a valid refresh token (sent via HttpOnly cookie). Returns a new access token.
* **Request Body:** None required (uses `refresh_token_cookie`).
* **Success Response (200 OK):**
    ```json
    {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." // New access token
    }
    ```
    *(Backend also sets a new `access_token_cookie`)*
* **Error Responses:**
    * `401 Unauthorized`: Invalid or missing refresh token cookie.
    * `422 Unprocessable Entity`: If CSRF check fails (when `JWT_REFRESH_COOKIE_CSRF_PROTECT=True`).
* **Authentication:** Valid `refresh_token_cookie` required.
* **Rate Limit:** 5 per minute.

#### `POST /api/auth/logout`

* **Description:** Clears the JWT access and refresh cookies.
* **Request Body:** None required.
* **Success Response (200 OK):**
    ```json
    {
      "message": "Logout successful"
    }
    ```
* **Authentication:** None required (but effectively logs out the current session).

#### `GET /api/auth/me`

* **Description:** Returns the identity (username) and role of the currently authenticated user based on their valid access token (header or cookie).
* **Success Response (200 OK):**
    ```json
    {
      "logged_in_as": "testuser",
      "role": "user"
    }
    ```
* **Error Responses:**
    * `401 Unauthorized`: Missing or invalid access token.
* **Authentication:** Valid access token required.
* **Rate Limit:** 120 per minute.

---

### Dashboard & Core Data

#### `GET /api/dashboard`

* **Description:** Retrieves aggregated data for the main dashboard view, including latest devices, recent alerts, overview metrics, and performance data.
* **Success Response (200 OK):**
    ```json
    {
      "systemHealth": 85,
      "activeDevices": 23,
      "totalDevices": 25,
      "efficiency": 91,
      "energyUsage": 123.4,
      "energyCost": 355,
      "statusDistribution": { "normal": 20, "warning": 2, "critical": 1, "offline": 2 },
      "timestamp": "...",
      "devices": [ ... ], // Array of device objects (latest state)
      "alerts": [ ... ], // Array of recent alert objects
      "performanceData": [ // Array of time-series performance points
        { "timestamp": "...", "systemHealth": 84, "efficiency": 90, "energyUsage": 120.1 },
        ...
      ]
    }
    ```
* **Error Responses:**
    * `401 Unauthorized`.
    * `500 Internal Server Error`.
* **Authentication:** Required.
* **Rate Limit:** 60 per minute.

#### `GET /api/devices`

* **Description:** Retrieves a list of all registered devices with their latest known status and data.
* **Query Parameters:**
    * `limit` (integer, optional, default: 1000): Maximum number of devices to return.
* **Success Response (200 OK):**
    ```json
    [
      {
        "id": 1, // DB ID
        "device_id": "DEVICE_001",
        "device_type": "temperature_sensor",
        "device_name": "Boiler Temp Sensor 001",
        "timestamp": "...",
        "value": 75.5,
        "status": "normal",
        "health_score": 0.92,
        "efficiency_score": 0.88,
        "location": "Boiler Room 1",
        "unit": "°C",
        "metadata": {} // Optional extra data
      },
      ...
    ]
    ```
* **Error Responses:**
    * `401 Unauthorized`.
    * `500 Internal Server Error`.
* **Authentication:** Required.
* **Rate Limit:** 120 per minute.

#### `GET /api/devices/<string:device_id>`

* **Description:** Retrieves the latest data for a specific device.
* **Path Parameters:**
    * `device_id`: The unique ID of the device (e.g., `DEVICE_001`).
* **Success Response (200 OK):** (Single device object, format as in `GET /api/devices`)
* **Error Responses:**
    * `401 Unauthorized`.
    * `404 Not Found`: If device ID doesn't exist.
    * `500 Internal Server Error`.
* **Authentication:** Required.
* **Rate Limit:** 120 per minute.

#### `GET /api/alerts`

* **Description:** Retrieves a list of recent alerts.
* **Query Parameters:**
    * `limit` (integer, optional, default: 50): Maximum number of alerts.
    * `severity` (string, optional): Filter by severity ('info', 'warning', 'critical').
    * `acknowledged` (boolean string, optional): Filter by acknowledgment status ('true' or 'false').
* **Success Response (200 OK):**
    ```json
    [
      {
        "id": "alert_uuid_string",
        "device_id": "DEVICE_002",
        "rule_name": "temperature_critical",
        "severity": "critical",
        "message": "Temperature reached critical level: 98.2 °C",
        "timestamp": "...",
        "acknowledged": false,
        "resolved": false,
        "value": 98.2,
        "metadata": {}
      },
      ...
    ]
    ```
* **Error Responses:**
    * `401 Unauthorized`.
    * `500 Internal Server Error`.
* **Authentication:** Required.
* **Rate Limit:** 120 per minute.

#### `POST /api/alerts/acknowledge/<string:alert_id>`

* **Description:** Marks a specific alert as acknowledged by the current user.
* **Path Parameters:**
    * `alert_id`: The unique ID of the alert.
* **Request Body:** None.
* **Success Response (200 OK):**
    ```json
    {
      "message": "Alert acknowledged",
      "alert": { ... } // The updated alert object
    }
    ```
* **Error Responses:**
    * `401 Unauthorized`.
    * `404 Not Found`: If alert ID doesn't exist.
    * `500 Internal Server Error`.
* **Authentication:** Required.
* **Rate Limit:** 30 per minute.

---

### Analytics & AI Endpoints

#### `GET /api/predictions`

* **Description:** Retrieves anomaly and failure predictions for a specific device based on recent data.
* **Query Parameters:**
    * `device_id` (string, required): The ID of the device.
* **Success Response (200 OK):**
    ```json
    {
      "device_id": "DEVICE_001",
      "anomaly_prediction": { // Output from detect_anomalies
        "anomaly_count": 0,
        "anomaly_percentage": 0.0,
        "anomaly_indices": [],
        ...
      },
      "failure_prediction": { // Output from predict_failure
        "predictions": [0, 0, ...],
        "failure_probabilities": [0.05, 0.06, ...],
        "high_risk_count": 0,
        ...
      }
    }
    ```
* **Error Responses:**
    * `400 Bad Request`: Missing `device_id`.
    * `401 Unauthorized`.
    * `404 Not Found`: If device has no recent data.
    * `500 Internal Server Error`.
    * `501 Not Implemented`: If the analytics engine is unavailable.
* **Authentication:** Required.
* **Rate Limit:** 60 per minute.

#### `GET /api/health_scores`

* **Description:** Retrieves the latest calculated health scores for all devices (or a subset).
* **Success Response (200 OK):**
    ```json
    {
      "DEVICE_001": 0.92, // Overall health score
      "DEVICE_002": 0.75,
      "DEVICE_003": 0.45
      // Potentially more details per device could be added
    }
    ```
* **Error Responses:**
    * `401 Unauthorized`.
    * `500 Internal Server Error`.
    * `501 Not Implemented`: If the health calculator is unavailable.
* **Authentication:** Required.
* **Rate Limit:** 60 per minute.

#### `GET /api/recommendations`

* **Description:** Retrieves AI-generated recommendations (maintenance, optimization, etc.).
* **Success Response (200 OK):**
    ```json
    [ // Flattened list, sorted by score/priority
      {
        "type": "maintenance",
        "priority_level": "high",
        "title": "Vibration Trend Increasing",
        "action": "Investigate vibration sensor on DEVICE_003",
        "urgency": 0.7,
        "impact": 0.8,
        "composite_score": 0.75,
        "timeframe": "within_week",
        "source": "historical_analysis"
      },
      ...
    ]
    ```
* **Error Responses:**
    * `401 Unauthorized`.
    * `500 Internal Server Error`.
    * `501 Not Implemented`: If the recommendation engine is unavailable.
* **Authentication:** Required.
* **Rate Limit:** 30 per minute.

---

### System & Reporting Endpoints

#### `GET /api/system_metrics`

* **Description:** Retrieves current system performance metrics (CPU, memory, etc.).
* **Success Response (200 OK):**
    ```json
    {
      "timestamp": "...",
      "cpu_percent": 15.2,
      "memory_percent": 65.8,
      "disk_percent": 72.1,
      "active_connections": 5, // WebSocket connections
      "cache_available": true,
      "database_available": true,
      "celery_available": true
    }
    ```
* **Error Responses:**
    * `401 Unauthorized`.
    * `500 Internal Server Error`.
* **Authentication:** Required.
* **Rate Limit:** 120 per minute.

#### `GET /api/historical_data`

* **Description:** Retrieves historical time-series data for a specific device.
* **Query Parameters:**
    * `device_id` (string, required): The device ID.
    * `hours` (integer, optional, default: 24): How many hours of data to retrieve.
    * `metric` (string, optional): Specific metric to retrieve (e.g., 'temperature'). Defaults to primary 'value'.
* **Success Response (200 OK):**
    ```json
    {
      "device_id": "DEVICE_001",
      "metric": "value",
      "unit": "°C",
      "timestamps": ["...", "...", ...],
      "values": [75.1, 75.3, ...]
    }
    ```
* **Error Responses:**
    * `400 Bad Request`: Missing `device_id`.
    * `401 Unauthorized`.
    * `500 Internal Server Error`.
* **Authentication:** Required.
* **Rate Limit:** 60 per minute.

#### `POST /api/tasks/start_report`

* **Description:** Starts an asynchronous task to generate a health report. (Admin only)
* **Success Response (202 Accepted):**
    ```json
    {
      "task_id": "celery_task_uuid",
      "status": "PENDING"
    }
    ```
* **Error Responses:**
    * `401 Unauthorized`.
    * `403 Forbidden`: If user is not an admin.
    * `500 Internal Server Error`.
    * `503 Service Unavailable`: If Celery is not configured/running.
* **Authentication:** Required (Admin Role).
* **Rate Limit:** 5 per hour.

#### `GET /api/tasks/status/<string:task_id>`

* **Description:** Checks the status of an asynchronous task (like report generation).
* **Path Parameters:**
    * `task_id`: The ID returned by the task initiation endpoint.
* **Success Response (200 OK):**
    * **Pending/Started:**
        ```json
        { "task_id": "...", "status": "PENDING" }
        ```
    * **Success:**
        ```json
        {
          "task_id": "...",
          "status": "SUCCESS",
          "result": { "status": "SUCCESS", "report_path": "/reports/health_report_..." }
        }
        ```
    * **Failure:**
        ```json
        { "task_id": "...", "status": "FAILURE", "error": "Error message from task" }
        ```
* **Error Responses:**
    * `401 Unauthorized`.
    * `404 Not Found`: If task ID is invalid.
    * `500 Internal Server Error`.
    * `503 Service Unavailable`: If Celery is not configured/running.
* **Authentication:** Required.
* **Rate Limit:** 60 per minute.

*(Note: The actual report file is served via `/reports/<filename>`)*

#### `GET /api/export_data`

* **Description:** Triggers an asynchronous task to export historical data.
* **Query Parameters:**
    * `format` (string, optional, default: 'json'): 'json' or 'csv'.
    * `days` (integer, optional, default: 7): Number of past days to export.
* **Success Response (202 Accepted):** Returns a task ID, similar to `/api/tasks/start_report`. Use `/api/tasks/status/<task_id>` to check progress and get the download path upon success. The result field will contain `export_path` and `filename`.
* **Error Responses:**
    * `401 Unauthorized`.
    * `500 Internal Server Error`.
    * `503 Service Unavailable`: If Celery is not configured/running.
* **Authentication:** Required.
* **Rate Limit:** 10 per hour.

*(Note: The actual export file is served via `/exports/<filename>`)*

---

## 2. WebSocket API (Flask-SocketIO)

**Connection:** Client connects to the main Flask server URL (e.g., `ws://localhost:5000` or `wss://yourdomain.com`).

**Authentication:** Requires sending a valid JWT access token during the connection handshake. This is typically done via the `auth` payload in the Socket.IO client connection options or automatically via the `access_token_cookie` if set.

```javascript
// Example Client Connection (using Header/Query Param - less common for web)
// const token = localStorage.getItem('jwt_token');
// const socket = io({ auth: { token } });

// Example Client Connection (relying on HttpOnly Cookie - preferred for web)
const socket = io({
  withCredentials: true // Crucial for sending cookies
});

socket.on('connect', () => { console.log('WebSocket Connected:', socket.id); });
socket.on('disconnect', (reason) => { console.log('WebSocket Disconnected:', reason); });
socket.on('connect_error', (error) => { console.error('WebSocket Connection Error:', error); });