#### 12. New File: `ARCHITECTURE.md`

```markdown
# Digital Twin System Architecture

This document outlines the high-level architecture of the Industrial IoT Digital Twin system.

## System Overview

The system is a full-stack web application designed for real-time monitoring, analysis, and prediction of industrial IoT device health. It consists of three main parts:
1.  **Backend (Flask)**: A Python-based application server that handles business logic, data processing, AI, and API requests.
2.  **Frontend (React)**: A browser-based single-page application (SPA) that provides the user interface, data visualization, and real-time updates.
3.  **Data Layer (SQLite)**: A set of file-based databases for storing device data, user credentials, and application state.

## Data Flow

1.  **Simulation**: `UnifiedDataGenerator` runs as a background task, creating realistic device data (including anomalies and trends) every 5 seconds.
2.  **Caching**: The latest data snapshot is fetched by the `DigitalTwinApp` and stored in an in-memory `data_cache`.
3.  **Real-time Broadcast (Push)**: The new data snapshot is immediately broadcast via **Socket.IO** to all connected frontend clients.
4.  **Client-Side Update**: The React frontend (`Dashboard.tsx`) listens for the `data_update` WebSocket event and updates its state, causing the UI to re-render with live data.
5.  **API Interaction (Pull)**:
    * A user logs in via the React `LoginPage`, which sends credentials to the Flask `/api/login` endpoint.
    * The backend validates credentials against the `users.db` and returns a **JWT**.
    * The React app stores this JWT in `localStorage`.
    * When the user navigates to the dashboard or clicks a button (e.g., "Get Recommendations"), the frontend makes an authenticated `fetch` call (e.g., to `/api/recommendations`) using the JWT.
    * The Flask backend (`enhanced_flask_app_v2.py`) receives the request, verifies the JWT, and calls the appropriate AI module (e.g., `RecommendationEngine`).
    * The AI module (e.g., `HealthScoreCalculator`) may pull historical data from `health_data.db` via the `SecureDatabaseManager`.
    * The AI module processes the data and returns the result (e.g., JSON recommendations) to the Flask app.
    * The Flask app serializes the result and sends it back to the React frontend as a JSON response.
    * The React component updates its state with the response and displays the new information.
6.  **Scheduled Tasks**: An `APScheduler` instance runs in the Flask backend, triggering the `analytics_engine.retrain_models` function every 24 hours. This function pulls fresh data from the `SecureDatabaseManager` to retrain and save new AI models.

## Component Diagram


* **User (Browser)** -> **React Frontend (SPA)**
    * -> **[WebSocket]** -> **Flask-SocketIO** (for real-time data)
    * -> **[HTTPS/API]** -> **Flask Backend (Gunicorn)** (for on-demand actions)
* **Flask Backend**
    * -> **JWTManager** (handles /login, /register, token verification)
    * -> **API Endpoints** (`/api/*`)
    * -> **AI Modules**
        * `PredictiveAnalyticsEngine` (reads/writes models from `ANALYTICS/models/`)
        * `HealthScoreCalculator`
        * `PatternAnalyzer`
        * `RecommendationEngine`
        * `AlertManager`
    * -> **Data Access Layer**
        * `SecureDatabaseManager` (handles encryption/decryption)
    * -> **Databases**
        * `users.db` (for auth)
        * `health_data.db` (for sensor data)
    * -> **Background Services**
        * `APScheduler` (triggers retraining)
        * `UnifiedDataGenerator` (simulates live data)