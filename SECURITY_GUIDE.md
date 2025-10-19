# Digital Twin Security Guide

This document outlines the security mechanisms implemented in the Digital Twin system.

## 1. Authentication

Authentication is handled via **JSON Web Tokens (JWT)**.

* **Flow**:
    1.  A user submits `username` and `password` to the `POST /api/login` endpoint.
    2.  The backend verifies these credentials against hashed passwords in the `users.db` using `werkzeug.security.check_password_hash`.
    3.  On success, the backend generates a signed JWT (using `flask-jwt-extended`) and returns it to the client.
    4.  The client (React app) stores this token in `localStorage`.
* **Protection**:
    * All API endpoints under `/api/*` (except `/login` and `/register`) are protected and require a valid JWT.
    * This is enforced by the `@jwt_required()` decorator in `enhanced_flask_app_v2.py`.
    * If no token or an invalid/expired token is provided, the API returns a `401 Unauthorized` error.
    * The frontend API client (`api.ts`) intercepts 401 errors and automatically redirects the user to the `/login` page.
* **Token Expiry**: Access tokens are configured to expire after **24 hours**.

## 2. Data Encryption (At-Rest)

Sensitive data in the database can be encrypted using symmetric encryption.

* **Algorithm**: Fernet (which uses AES128-CBC with a 128-bit key for encryption, and HMAC with SHA256 for authentication).
* **Key Management**:
    * A unique, secret encryption key is stored in `Digital_Twin/CONFIG/encryption.key`.
    * **This file MUST NOT be committed to version control.** It must be generated and placed on the server manually.
    * The `SecureDatabaseManager` loads this key on initialization.
* **Implementation**:
    * The `secure_database_manager.py` provides `encrypt_data(data)` and `decrypt_data(encrypted_data)` methods.
    * These methods are used to encrypt specific columns (e.g., PII, sensitive configs) before writing to the database and decrypt them after reading.

## 3. Password Hashing

User passwords are **never** stored in plain text.

* **Algorithm**: `sha256` with a per-user salt.
* **Implementation**: We use `werkzeug.security.generate_password_hash` to create hashes on registration and `check_password_hash` to verify them on login. This protects against rainbow table attacks.

## 4. Environment Variables

Sensitive configuration, such as the `SECRET_KEY` and `JWT_SECRET_KEY`, should be set using environment variables in a production environment, not hardcoded. The app is designed to read from `os.environ.get()`.

## 5. WebSockets

WebSocket connections are initiated by the authenticated frontend. While the current implementation allows connections from `*`, in a production environment, `CORS_ALLOWED_ORIGINS` and `cors_allowed_origins` for Flask and SocketIO should be restricted to the specific frontend domain.