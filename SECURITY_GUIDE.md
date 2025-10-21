***

## `SECURITY_GUIDE.md`

The existing guide is quite comprehensive. I've reviewed it and added minor clarifications regarding AES-GCM and explicitly mentioned key rotation practices.

```markdown
# Digital Twin Security Guide v2.1

This document outlines the security mechanisms implemented in the Digital Twin system.

## 1. Authentication

Authentication is handled via **JSON Web Tokens (JWT)**, managed by `flask-jwt-extended`.

* **Flow**:
    1.  User submits credentials (`username`, `password`) to `POST /api/auth/login`.
    2.  Backend verifies credentials against **bcrypt** hashed passwords (see Password Hashing section).
    3.  On success, the backend generates a signed **access token** (short-lived) and a **refresh token** (longer-lived).
    4.  The access token is returned in the response body. Both access and refresh tokens are also set as **HttpOnly, Secure cookies** (in production). The refresh token cookie is restricted to the `/api/auth/refresh` path.
* **Protection**:
    * Protected API endpoints require a valid access token, checked via the `Authorization: Bearer <token>` header or the `access_token_cookie`. Enforced by `@jwt_required()`.
    * The `/api/auth/refresh` endpoint requires a valid `refresh_token_cookie`. Enforced by `@jwt_required(refresh=True)`.
    * Invalid/expired tokens result in `401 Unauthorized` or `422 Unprocessable Entity` (if CSRF check fails).
    * Frontend should handle 401s by attempting to refresh the token using the refresh endpoint or redirecting to login if refresh fails.
* **Token Expiry & Refresh**:
    * Access token expiry (`JWT_ACCESS_TOKEN_EXPIRES`) is short (e.g., 1 hour) for better security.
    * Refresh token expiry (`JWT_REFRESH_TOKEN_EXPIRES`) is longer (e.g., 30 days) for user convenience.
    * Clients use the refresh token (via the cookie sent automatically to `/api/auth/refresh`) to obtain a new access token without requiring re-login.
* **CSRF Protection with Cookies**: `flask-jwt-extended` is configured with `JWT_COOKIE_CSRF_PROTECT=True`. This requires clients using cookie-based authentication to include a CSRF token (obtained separately, often via a dedicated endpoint or initial page load) in requests, typically in the `X-CSRF-TOKEN` header. This mitigates CSRF attacks when using cookies for JWT storage.

## 2. Data Encryption (At-Rest)

Sensitive data stored in the database (managed via `secure_database_manager.py` using SQLAlchemy) is encrypted using **AES-256-GCM**.

* **Algorithm**: AES-256 in **Galois/Counter Mode (GCM)**.
    * **Confidentiality:** Ensures data cannot be read without the key.
    * **Integrity & Authenticity:** GCM includes an **authentication tag**. This tag verifies that the data has **not been tampered with** since it was encrypted. If the ciphertext or nonce is altered, decryption will fail.
* **Key Management**:
    * A **32-byte (256-bit) secret encryption key** is required.
    * This key **MUST** be provided via the `ENCRYPTION_KEY` environment variable.
    * The key must be **base64-encoded** when set in the environment variable.
    * Use `python Digital_Twin/AI_MODULES/secure_database_manager.py generate_key` to create a suitable key.
    * **Secret Management:** Store the key securely (environment variables, secret manager). **NEVER commit the key to version control.**
* **Implementation**:
    * Uses the `cryptography` library within `secure_database_manager.py`.
    * `encrypt_data(data)`: Encrypts data, adding a unique nonce (12 bytes) and the GCM authentication tag. Stores `base64(nonce + ciphertext + tag)`.
    * `decrypt_data(encrypted_data)`: Extracts nonce, ciphertext, and tag. Decrypts and **verifies the authentication tag**. Raises `InvalidTag` if verification fails, preventing use of tampered data.
* **Key Rotation:**
    * **Procedure:** Generate a new `ENCRYPTION_KEY`. Update the environment variable for the application. Restart the application.
    * **Considerations:** Data encrypted with the *old key* will no longer be decryptable unless you implement a mechanism to handle multiple keys (e.g., storing a key ID with the ciphertext and keeping old keys available for decryption). For simpler systems, key rotation might involve a planned data migration step where data is decrypted with the old key and re-encrypted with the new key.
    * **Frequency:** Rotate keys periodically (e.g., annually or based on compliance requirements) or immediately if a key is suspected to be compromised.

## 3. Password Hashing

User passwords are **never** stored in plain text.

* **Algorithm**: **bcrypt** is used. It's a strong, adaptive hashing algorithm with integrated salting, resistant to brute-force and rainbow table attacks.
* **Implementation**: Uses the `bcrypt` library. Passwords are hashed on registration (`bcrypt.hashpw`) and verified on login (`bcrypt.checkpw`).

## 4. Environment Variables & Secret Management

* **Principle**: ALL sensitive configuration (database URLs, API keys, secret keys (`SECRET_KEY`, `JWT_SECRET_KEY`), `ENCRYPTION_KEY`, SMTP passwords, etc.) **MUST** be loaded from **environment variables** or a dedicated secret management system (like HashiCorp Vault, AWS Secrets Manager, etc.).
* **Loading**: The application uses helper functions (`get_required_env`, `get_optional_env`) to load these at runtime.
* **`.env` File**: Should **only be used for local development** and **MUST be added to `.gitignore`**. Do not commit real secrets.
* **`.env.example`**: A template file (containing placeholders, not real secrets) should be committed as a guide.
* **Production**: Use the environment management system of your deployment platform (Docker secrets, Kubernetes secrets, cloud provider secret managers, OS environment variables).
* **Configuration Files**: Files like `alert_config.json` should reference environment variables (e.g., `env_SMTP_PASSWORD`) for secrets, which are resolved by the application during loading. Avoid hardcoding credentials.

## 5. API & WebSocket Security

* **WebSocket Authentication**: Connections require a valid JWT passed during the handshake (via cookie or `auth` payload). Unauthorized connections are rejected.
* **API Rate Limiting**: Flask-Limiter is implemented (configured via `RATELIMIT_*` env vars, using Redis) to prevent abuse.
* **CORS**: Configured via `CORS_ALLOWED_ORIGINS`, restricting browser access to specified frontend domains in production. `supports_credentials=True` allows cookies to be sent/received.
* **Input Validation**: Marshmallow schemas (`RegisterSchema`, `LoginSchema`) are used for strict API input validation to prevent injection and malformed data issues.
* **CSRF Protection**:
    * **Forms**: Enabled via `Flask-WTF` (`WTF_CSRF_ENABLED=True`) for traditional form submissions.
    * **JWT Cookies**: `flask-jwt-extended` (`JWT_COOKIE_CSRF_PROTECT=True`) provides "double submit" cookie protection, requiring a CSRF token in requests using JWT cookies.
    * **API Headers**: For header-based JWT (`Authorization: Bearer`), CSRF is less of a direct risk, but the frontend must be protected against XSS to prevent token theft.
* **HTTPS**: Essential for production. Nginx (or another reverse proxy) handles HTTPS termination, redirects HTTP->HTTPS, and employs security headers (HSTS, X-Frame-Options, etc.). See `nginx.conf` for examples.
* **Role-Based Access Control (RBAC)**: Implemented using custom decorators (`@role_required`) checking the `role` claim within the JWT.

## 6. Dependency Management

* Regularly update dependencies (`requirements.txt`, `package.json`).
* Scan for known vulnerabilities using tools like `pip-audit`, `safety` (Python), or `npm audit` (Node.js).

## 7. Logging & Monitoring

* **Structured Logging**: `structlog` provides detailed, queryable logs (JSON format configurable via `LOG_FORMAT` env var).
* **Audit Logging**: `secure_database_manager.py` logs sensitive operations (auth events, data modifications, integrity failures) to a dedicated audit trail.
* **Monitoring**: Prometheus metrics are exposed via `/metrics` for application performance monitoring. Integrate with Grafana for visualization.

## 8. Best Practices Summary

* **Secrets Management**: Centralize secrets in environment variables or a vault. Never commit secrets. Rotate keys regularly.
* **HTTPS**: Enforce HTTPS in production.
* **Input Validation**: Validate and sanitize all user/external inputs.
* **Rate Limiting**: Protect APIs from abuse.
* **Dependency Scanning**: Keep libraries updated and vulnerability-free.
* **Secure Headers**: Implement CSP, HSTS, X-Frame-Options, etc., via the reverse proxy.
* **Least Privilege**: Run processes and database users with minimal necessary permissions.
* **Regular Audits**: Periodically review logs, configurations, and access controls.

---
**Next Steps:**

* **Scrub Git History:** If secrets were *ever* committed, remove them completely from history.
* **Implement Key Rotation Strategy:** Define and document the process for rotating `ENCRYPTION_KEY`, `SECRET_KEY`, and `JWT_SECRET_KEY`.