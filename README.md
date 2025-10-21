# 🏭 Digital Twin System v2

A comprehensive Digital Twin platform for industrial IoT monitoring, predictive analytics, and real-time system health management. Built with Flask, React, Celery, and various monitoring/data tools.

## 🌟 Features

* **Real-time Monitoring**: Live data streaming via WebSockets and MQTT.
* **Predictive Analytics**: AI-powered anomaly detection, failure prediction, and pattern analysis.
* **Interactive Dashboard**: React-based web interface with real-time visualizations (powered by `Recharts`).
* **Health Scoring**: Automated system and device health assessment.
* **Alert Management**: Intelligent alerting with configurable thresholds and notifications (Email, Slack, Webhooks).
* **Secure & Scalable**: Uses JWT authentication, AES-GCM encryption (at-rest), and a containerized architecture with Docker Compose.
* **Background Tasks**: Celery for asynchronous operations and scheduled tasks (like model retraining).
* **Monitoring**: Integrated Prometheus for metrics collection and Grafana for visualization.
* **Demo Mode**: Option to run the frontend with simulated data without a live backend.

## 🏗️ Architecture Overview

The system consists of several key components working together:

1.  **IoT Devices (Simulated/Real):** Send data via MQTT.
2.  **MQTT Broker (Mosquitto):** Relays messages from devices.
3.  **MQTT Ingestor (Python Service):** Subscribes to MQTT topics, parses data, and stores it in the database.
4.  **Database (PostgreSQL):** Stores device data, historical metrics, alerts, user info, etc.
5.  **Backend (Flask):**
    * Exposes a REST API for data retrieval and actions.
    * Manages WebSocket connections for real-time UI updates.
    * Handles user authentication (JWT).
    * Coordinates with AI Modules and Celery.
6.  **AI Modules (Python):** Perform calculations for health scores, alerts, predictions, patterns, and recommendations.
7.  **Real-time Layer (Flask-SocketIO + Redis):** Pushes live updates to connected clients.
8.  **Task Queue (Celery + Redis):** Handles background tasks like report generation, model retraining, and notifications.
9.  **Cache (Redis):** Caches frequently accessed data to improve performance.
10. **Frontend (React/Vite):** Single-page application for user interaction and data visualization.
11. **Monitoring (Prometheus + Grafana):** Collects metrics from the backend and provides dashboards for system monitoring.
12. **Reverse Proxy (Nginx):** Manages incoming traffic, handles HTTPS, and routes requests to appropriate services (Flask, Grafana, Prometheus).

**(For a visual diagram, please refer to the `ARCHITECTURE.md` file or generate one based on the description above).**

## 🚀 Quick Start Guide (Using Docker Compose)

This is the recommended way to run the entire system locally.

### Prerequisites

* Docker & Docker Compose installed.
* Git installed.
* A `.env` file created in the `Digital_Twin` directory (see `.env.example` or the provided `.env` file for required variables). **Make sure to generate and fill in strong, unique secrets!**

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd updated_digital # Navigate to the root directory containing Digital_Twin and the React app src
    ```

2.  **Navigate to the Backend Directory:**
    ```bash
    cd Digital_Twin
    ```

3.  **Create and Configure `.env`:**
    * Copy the provided `.env` file or `.env.example` to `.env`.
    * **Crucially, generate unique secrets** for `SECRET_KEY`, `JWT_SECRET_KEY`, and `ENCRYPTION_KEY`. Use `python AI_MODULES/secure_database_manager.py generate_key` for the `ENCRYPTION_KEY` (copy the base64 output). Use commands like `openssl rand -hex 32` for the others.
    * Fill in database passwords (`POSTGRES_PASSWORD`), Grafana admin password (`GF_SECURITY_ADMIN_PASSWORD`), etc. Ensure `POSTGRES_PASSWORD` matches the password in `DATABASE_URL`.
    * Set `DOMAIN_NAME` (e.g., `localhost` for local testing).

4.  **Build and Start Services:**
    ```bash
    docker compose up --build -d
    ```
    *(The `--build` flag ensures images are built based on the Dockerfile. `-d` runs services in detached mode.)*

5.  **Access Services:**
    * **Main Application:** `http://localhost` (or your `DOMAIN_NAME`). Nginx handles routing.
    * **Grafana:** `http://localhost/grafana/` (Login with `admin` and the password set in `.env`).
    * **Prometheus:** `http://localhost/prometheus/`
    * **(Optional) MQTT Broker:** Ports 1883 (MQTT) and 9001 (WebSockets) are exposed if you need direct MQTT access.

6.  **Stopping Services:**
    ```bash
    docker compose down
    ```

## 📊 Monitoring Setup (Prometheus + Grafana)

The `docker-compose.yml` sets up Prometheus and Grafana services.

* **Prometheus:** Scrapes metrics exposed by the Flask backend at its internal `/metrics` endpoint. Configuration is in `prometheus.yml`.
* **Grafana:** Visualizes data collected by Prometheus.
    * **Access:** `http://localhost/grafana/`
    * **Login:** `admin` / `<Your GF_SECURITY_ADMIN_PASSWORD from .env>`
    * **Data Source:** A Prometheus data source should be pre-configured via provisioning (`grafana/provisioning/datasources`).
    * **Dashboards:** Example dashboards can be added to `grafana/provisioning/dashboards` to be automatically imported.

## 💡 Demo Mode (Frontend)

The React frontend includes a "Demo Mode" option on the login screen.

* **Functionality:** When enabled, the frontend uses **local simulated data** generated within the browser (`src/utils/data-simulator.ts`) and stored in **IndexedDB** (`src/utils/db.ts`). It **does not** communicate with the live backend API or WebSockets.
* **Purpose:** Allows exploring the UI and features without needing the full backend stack running or requiring login credentials. Useful for quick UI checks or offline demonstrations.
* **Activation:** Toggle the "Demo Mode" switch on the login page (`src/pages/Login.tsx`). The state is stored in `localStorage`.
* **Limitations:** Data is simulated and ephemeral. No real-time updates from external sources, no interactions with the actual AI modules, alerts, or database occur.

## 📚 Documentation

* [API Documentation](API_DOCUMENTATION.md)
* [Architecture Overview](ARCHITECTURE.md)
* [Security Guide](SECURITY_GUIDE.md)
* *(Additional guides like Deployment, Maintenance can be added)*

## 🤝 Contributing

*(Add contribution guidelines here if applicable)*

## 📄 License

*(Specify your project's license, e.g., MIT License)*