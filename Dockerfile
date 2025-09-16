# Use Python 3.11 slim as base image
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_DISABLE_CACHING=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libopencv-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Playwright dependencies removed to avoid build issues

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Skip Playwright browser installation to avoid Node.js dependency issues
# Browsers can be installed at runtime if needed
RUN echo 'Skipping Playwright browser installation to avoid build issues'

# Copy the application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p logs data cache /tmp/numba_cache /tmp/matplotlib && \
    chmod -R 777 /tmp/numba_cache /tmp/matplotlib /tmp

# Set proper permissions
RUN chmod +x main.py main_minimal.py

# Create a non-root user for security (skip if already exists)
RUN (groupadd -r iram 2>/dev/null || true) && \
    (useradd -r -g iram iram 2>/dev/null || true) && \
    chown -R iram:iram /app

# Switch to non-root user (temporarily disabled to simplify debugging)
# USER iram

# Expose port
EXPOSE 8000

# Health check to probe dynamic port (use /health which exists in all modes)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

# Default command - run full MCP server via uvicorn (respects dynamic PORT)
CMD ["sh", "-c", "uvicorn src.mcp_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
