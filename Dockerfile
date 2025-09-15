# Use Python 3.12 slim as base image
FROM python:3.12-slim

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

# Install Playwright dependencies
RUN apt-get update && apt-get install -y \
    libnss3-dev \
    libatk-bridge2.0-dev \
    libdrm-dev \
    libxcomposite-dev \
    libxdamage-dev \
    libxrandr-dev \
    libgbm-dev \
    libxss-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium

# Copy the application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p logs data cache /tmp/numba_cache /tmp/matplotlib && \
    chmod -R 777 /tmp/numba_cache /tmp/matplotlib /tmp

# Set proper permissions
RUN chmod +x main.py

# Create a non-root user for security
RUN groupadd -r iram && useradd -r -g iram iram && \
    chown -R iram:iram /app

# Switch to non-root user
USER iram

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py", "server"]