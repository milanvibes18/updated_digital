# Digital Twin Security Guide v2

This document outlines the security mechanisms implemented in the Digital Twin system.

## 1. Authentication

Authentication is handled via **JSON Web Tokens (JWT)**.

* **Flow**:
    1.  User submits credentials to `POST /api/auth/login`.
    2.  Backend verifies against hashed passwords using `werkzeug.security.check_password_hash`.
    3.  On success, backend generates a signed JWT (using `flask-jwt-extended`).
    4.  The token is returned to the client and can be stored (e.g., in memory, session storage) or the backend sets an **HttpOnly, Secure cookie** containing the JWT.
* **Protection**:
    * Protected API endpoints require a valid JWT passed via `Authorization: Bearer <token>` header or via the secure cookie.
    * Enforced by `@jwt_required()` decorator.
    * Invalid/expired tokens result in `401 Unauthorized`.
    * Frontend should handle 401s by redirecting to login.
* **Token Expiry**: Configurable via `JWT_EXPIRY_HOURS` environment variable (default 24 hours).
* **CSRF Protection**: If using cookie-based JWTs, Flask-JWT-Extended provides CSRF protection options (`JWT_COOKIE_CSRF_PROTECT`).

## 2. Data Encryption (At-Rest)

Sensitive data stored in the database (e.g., in `secure_database_manager.py`) is encrypted using **AES-256-GCM**.

* **Algorithm**: AES-256 in Galois/Counter Mode (GCM). This provides **authenticated encryption**, ensuring both confidentiality and integrity (protection against tampering).
* **Key Management**:
    * A **32-byte (256-bit) secret encryption key** is required.
    * This key **MUST** be provided via the `ENCRYPTION_KEY` environment variable.
    * The key must be **base64-encoded** when set in the environment variable.
    * Use the `secure_database_manager.py generate_key` command to create a suitable key.
    * **NEVER commit the key to version control.**
* **Implementation**:
    * `secure_database_manager.py` uses the `cryptography` library.
    * `encrypt_data(data)`: Encrypts data, adding a unique nonce and an authentication tag. Stores nonce+ciphertext+tag, base64-encoded.
    * `decrypt_data(encrypted_data)`: Decrypts data. **Crucially, it verifies the authentication tag.** If the data has been tampered with, decryption will fail with an `InvalidTag` error, preventing the use of corrupted data.

## 3. Password Hashing

User passwords are **never** stored in plain text.

* **Algorithm**: `scrypt` or `pbkdf2_sha256` via `werkzeug.security`. These are strong, salted hashing algorithms resistant to brute-force and rainbow table attacks.
* **Implementation**: `generate_password_hash` on registration, `check_password_hash` on login.

## 4. Environment Variables & Configuration

* **ALL** sensitive configuration values (database URLs, API keys, secret keys, encryption keys, SMTP passwords, etc.) **MUST** be loaded from **environment variables**.
* The application uses helper functions (`get_required_env`, `get_optional_env`) to load these variables at runtime.
* A `.env` file can be used for local development (ensure it's in `.gitignore`). In production, use the environment management system of your deployment platform (e.g., Docker secrets, Kubernetes secrets, system environment variables).
* Configuration files like `alert_config.json` should contain placeholders (e.g., `env_SMTP_PASSWORD`) where secrets are expected, and the application code should resolve these placeholders against environment variables.

## 5. WebSockets & API Security

* **WebSocket Authentication**: Connections require a valid JWT passed during the connection handshake. Unauthorized connections are rejected.
* **API Rate Limiting**: Flask-Limiter is implemented (configured via environment variables, using Redis) to prevent abuse of API endpoints. Limits are applied globally and can be customized per-route.
* **CORS**: Cross-Origin Resource Sharing is configured via `CORS_ALLOWED_ORIGINS` environment variable, restricting access to specified frontend domains in production.
* **Input Validation**: (Medium Priority Item 9) Implementing libraries like Marshmallow or Pydantic for strict API input schema validation is crucial to prevent injection attacks and malformed data issues.
* **CSRF Protection**: Enabled for JWT cookies if that method is used. For header-based JWT, CSRF is less of a direct concern but ensure frontend protects against credential leakage.
* **HTTPS**: Nginx (or another reverse proxy) should be configured to handle HTTPS termination, redirect HTTP to HTTPS, and employ security headers (HSTS, X-Frame-Options, etc.). The provided `nginx.conf` includes examples.

## 6. Dependency Management

* Regularly update dependencies (`requirements.txt`, `package.json`) and scan for known vulnerabilities using tools like `pip-audit` or `npm audit`.

## 7. Logging & Monitoring

* Comprehensive logging captures key events, including authentication successes/failures and potential errors (see `setup_logging`).
* Audit logging (via `SecureDatabaseManager`) tracks sensitive operations.
* Integrate with monitoring tools (Prometheus/Grafana planned) to observe application behavior and detect anomalies.

---
**Next Steps:**

* **Scrub Git History:** If secrets were *ever* committed, they must be removed from the Git history using tools like `git filter-repo` or BFG Repo-Cleaner. Simply removing them in a new commit is not enough.
* **Implement Placeholder Resolution:** Update `alert_manager.py` (and potentially other modules reading config files) to replace `env_VAR_NAME` strings with actual environment variable values using `os.environ.get()`.
* **Generate Production Keys:** Create strong, unique keys/secrets for `SECRET_KEY`, `JWT_SECRET_KEY`, and `ENCRYPTION_KEY` for your production environment.