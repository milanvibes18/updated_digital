# Digital Twin Security Guide v2

This document outlines the security mechanisms implemented in the Digital Twin system.

## 1. Authentication

Authentication is handled via **JSON Web Tokens (JWT)**.

* **Flow**:
    1.  User submits credentials to `POST /api/auth/login`.
    2.  Backend verifies against hashed passwords (see Password Hashing section below).
    3.  On success, backend generates a signed JWT (using `flask-jwt-extended`).
    4.  The token is returned to the client and can be stored (e.g., in memory, local storage - *ensure protection against XSS*) OR the backend sets an **HttpOnly, Secure cookie** containing the JWT (preferred for web browsers).
* **Protection**:
    * Protected API endpoints require a valid JWT passed via `Authorization: Bearer <token>` header or via the secure cookie.
    * Enforced by `@jwt_required()` decorator.
    * Invalid/expired tokens result in `401 Unauthorized`.
    * Frontend should handle 401s by redirecting to login.
* **Token Expiry & Refresh**: Configurable via `JWT_ACCESS_TOKEN_EXPIRES`. `flask-jwt-extended` supports refresh tokens for managing longer sessions securely, although not explicitly implemented in the provided `enhanced_flask_app_v2.py`. Consider implementing refresh tokens for improved user experience.

## 2. Data Encryption (At-Rest)

Sensitive data stored in the database (e.g., in `secure_database_manager.py`) is encrypted using **AES-256-GCM**.

* **Algorithm**: AES-256 in Galois/Counter Mode (GCM). This provides **authenticated encryption**, ensuring both confidentiality and integrity (protection against tampering).
* **Key Management**:
    * A **32-byte (256-bit) secret encryption key** is required.
    * This key **MUST** be provided via the `ENCRYPTION_KEY` environment variable.
    * The key must be **base64-encoded** when set in the environment variable.
    * Use the `secure_database_manager.py generate_key` command to create a suitable key.
    * **NEVER commit the key to version control.** Store it securely (e.g., environment variables, secret manager).
* **Implementation**:
    * `secure_database_manager.py` uses the `cryptography` library.
    * `encrypt_data(data)`: Encrypts data, adding a unique nonce and an authentication tag. Stores nonce+ciphertext+tag, base64-encoded.
    * `decrypt_data(encrypted_data)`: Decrypts data. **Crucially, it verifies the authentication tag.** If the data has been tampered with, decryption will fail with an `InvalidTag` error, preventing the use of corrupted data.

## 3. Password Hashing

User passwords are **never** stored in plain text.

* **Algorithm**: **bcrypt** is recommended and implemented. It is a strong, adaptive hashing algorithm with integrated salting, resistant to brute-force and rainbow table attacks. (Previously used default `werkzeug.security` hash, now explicitly using `bcrypt`).
* **Implementation**: Uses the `bcrypt` library. Passwords are hashed on registration (`bcrypt.hashpw`) and verified on login (`bcrypt.checkpw`).

## 4. Environment Variables & Secret Management

* **ALL** sensitive configuration values (database URLs, API keys, secret keys, encryption keys, SMTP passwords, etc.) **MUST** be loaded from **environment variables** or a dedicated secret management system.
* The application uses helper functions (`get_required_env`, `get_optional_env`) to load these variables at runtime.
* A `.env` file **should only be used for local development** and **MUST be added to `.gitignore`**.
* A `.env.example` file (containing placeholders, not real secrets) should be committed to the repository as a template.
* In production, use the environment management system of your deployment platform (e.g., Docker secrets, Kubernetes secrets, cloud provider secret managers, system environment variables).
* Configuration files like `alert_config.json` should contain placeholders (e.g., `env_SMTP_PASSWORD`) where secrets are expected, and the application code should resolve these placeholders against environment variables during loading. Hardcoded credentials (even usernames) should be avoided.

## 5. WebSockets & API Security

* **WebSocket Authentication**: Connections require a valid JWT passed during the connection handshake. Unauthorized connections are rejected.
* **API Rate Limiting**: Flask-Limiter is implemented (configured via environment variables, using Redis) to prevent abuse of API endpoints. Limits are applied globally and can be customized per-route.
* **CORS**: Cross-Origin Resource Sharing is configured via `CORS_ALLOWED_ORIGINS` environment variable, restricting access to specified frontend domains in production.
* **Input Validation**: Libraries like Marshmallow are used for strict API input schema validation to prevent injection attacks and malformed data issues.
* **CSRF Protection**:
    * Enabled via `Flask-WTF`. This protects form submissions.
    * If using **cookie-based JWTs**, `flask-jwt-extended` (`JWT_COOKIE_CSRF_PROTECT`) provides additional protection by requiring CSRF tokens for cookie-authenticated requests, mitigating CSRF risks associated with cookies.
    * For **header-based JWTs** (`Authorization: Bearer`), CSRF is generally less of a direct risk for the API itself, as browsers don't automatically send this header on cross-origin requests. However, ensure the frontend application storing the token is protected against XSS.
* **HTTPS**: Essential for production. Nginx (or another reverse proxy) should be configured to handle HTTPS termination, redirect HTTP to HTTPS, and employ security headers (HSTS, X-Frame-Options, Content-Security-Policy, etc.). The provided `nginx.conf` includes examples.

## 6. Dependency Management

* Regularly update dependencies (`requirements.txt`, `package.json`) and scan for known vulnerabilities using tools like `pip-audit`, `safety`, or `npm audit`.

## 7. Logging & Monitoring

* **Structured Logging**: `structlog` is used for detailed, structured application logs, capturing request context, events, and errors.
* **Audit Logging**: `SecureDatabaseManager` logs sensitive operations (user creation, auth success/failure, data insertion, integrity failures) to a separate audit trail.
* **Monitoring**: Prometheus metrics are exposed via `/metrics` for monitoring application performance and behavior. Integrate with Grafana for visualization.

## 8. Best Practices Summary

* **Secrets Management**: Use environment variables or secret managers. Never hardcode secrets. Add `.env` to `.gitignore`.
* **HTTPS**: Enforce HTTPS for all communication in production.
* **Input Validation**: Validate and sanitize all incoming data (API requests, forms).
* **Rate Limiting**: Protect against brute-force and denial-of-service attacks.
* **Dependency Scanning**: Regularly check for vulnerable libraries.
* **Secure Headers**: Implement security headers (CSP, HSTS, X-Frame-Options, etc.) via the reverse proxy or Flask middleware.
* **Least Privilege**: Ensure database users and application processes run with the minimum necessary permissions.
* **Regular Audits**: Periodically review logs and security configurations.

---
**Next Steps:**

* **Scrub Git History:** If secrets were *ever* committed, they must be removed from the Git history using tools like `git filter-repo` or BFG Repo-Cleaner. Simply removing them in a new commit is not enough.
* **Review Dynamic Config Loading:** Ensure `alert_config.json` and similar files dynamically load secrets from environment variables as described.
* **Generate Production Keys:** Create strong, unique keys/secrets for `SECRET_KEY`, `JWT_SECRET_KEY`, and `ENCRYPTION_KEY` for your production environment and store them securely.