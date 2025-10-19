# System Architecture

## 1. Overview
This document outlines the high-level architecture of the Digital Twin IoT Monitoring platform. The system is designed as a real-time, data-driven application that connects a React-based frontend to a Python backend, supported by AI modules for analytics and alerts.

## 2. System Components

* **Frontend:** A React (Vite) single-page application (SPA) responsible for all data visualization and user interaction. It communicates with the backend via REST API and WebSockets.
* **Backend (Web Application):** A Flask server that exposes RESTful API endpoints for data retrieval (devices, alerts) and user actions (acknowledging alerts). It also manages WebSocket connections for real-time updates.
* **AI Modules (Digital_Twin/AI_MODULES):** A collection of Python scripts responsible for the core business logic:
    * `health_score.py`: Calculates device health.
    * `predictive_analytics_engine.py`: Predicts failures.
    * `alert_manager.py`: Manages alert generation based on thresholds.
    * `pattern_analyzer.py`: Analyzes data for patterns.
* **Database:** A Blink-managed database (likely SQLite for development) storing device data, historical metrics, alerts, and user settings.
* **Real-time Layer:** A WebSocket manager (`realtime_websocket_manager.py`) that pushes live updates from the AI modules and database to all connected frontend clients.

## 3. Data Flow Diagram

Below is a conceptual data flow for a real-time update.

[Physical IoT Device] | (Data Ingestion) | v [Flask Backend API] | v [Database (Blink)] <---> [AI Modules (Health, Alerts, etc.)] | | | v '----------------> [WebSocket Manager] | v [Frontend Client (React)] (Dashboard updated in real-time)


## 4. Frontend-AI Integration
1.  **Initial Load:** The React app loads and calls `GET /api/devices` and `GET /api/alerts` from the Flask backend to populate the dashboard.
2.  **Real-time Connection:** The React app establishes a WebSocket connection to the backend.
3.  **AI Processing:** As new data comes in, the AI modules process it, calculate new health scores, and check for alerts.
4.  **Push Update:** When a new alert is generated or a device status changes, the `alert_manager` or `health_score` module triggers the `realtime_websocket_manager`.
5.  **Broadcast:** The WebSocket manager broadcasts a JSON payload (e.g., `{'type': 'NEW_ALERT', 'data': {...}}`) to all connected frontend clients.
6.  **UI Update:** The React frontend (via the `useDashboard` hook) receives this message, updates its state, and re-renders the UI instantly, showing the new alert or device status.