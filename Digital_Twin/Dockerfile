# --- Stage 1: Build React Frontend ---
# Use a Node.js base image to build the React app
FROM node:20-alpine AS react-builder

WORKDIR /app-frontend

# Copy frontend package files and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the frontend source code
# These paths assume the Dockerfile is in the project root
COPY src /app-frontend/src
COPY public /app-frontend/public
COPY vite.config.ts .
COPY tsconfig.json .
COPY tsconfig.node.json .
COPY tailwind.config.js .
COPY postcss.config.cjs .
COPY components.json .

# Build the production-ready static files
RUN npm run build

# --- Stage 2: Build Python Backend Dependencies ---
# Use the same Python version as your old file
FROM python:3.12-slim AS python-builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed for building wheels (like numpy, cryptography)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app-backend

# Copy requirements and install Python dependencies
# This path assumes requirements.txt is inside the Digital_Twin folder
COPY Digital_Twin/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 3: Final Production Image ---
FROM python:3.12-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5000 \
    # Set the Python path to include the /app directory
    PYTHONPATH=/app

# Install runtime-only dependencies (like curl for healthcheck)
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (from your old, secure file)
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy Python dependencies from python-builder stage
COPY --from=python-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy the *built* frontend from react-builder stage into the Flask static folder
COPY --from=react-builder /app-frontend/dist /app/Digital_Twin/WEB_APPLICATION/static

# Copy backend application code
COPY Digital_Twin /app/Digital_Twin
COPY ANALYTICS /app/ANALYTICS

# Create necessary directories and set permissions (from your old file)
# Paths are adjusted to be relative to /app
RUN mkdir -p \
    /app/DATABASE \
    /app/LOGS \
    /app/REPORTS/generated \
    /app/ANALYTICS/models \
    /app/ANALYTICS/analysis_cache \
    # Note: Security-related directories should ideally be volumes
    /app/Digital_Twin/SECURITY/audit_logs \
    /app/Digital_Twin/SECURITY/data_backups \
    /app/Digital_Twin/CONFIG \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check (from your old file)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the application using the robust Gunicorn command (from your old file)
# BUT updated to use the app factory path (from your new file)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--timeout", "120", \
     "--keep-alive", "2", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "Digital_Twin.WEB_APPLICATION.enhanced_flask_app_v2:create_app()"]