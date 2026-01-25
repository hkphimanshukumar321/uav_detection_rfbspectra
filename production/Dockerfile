FROM python:3.9-slim-bullseye

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="DroneRFB-Spectra: Turnkey RF-based drone detection"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    librtlsdr-dev \
    rtl-sdr \
    libatlas-base-dev \
    libhdf5-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements_production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_production.txt

# Copy application
COPY edge_deployment/ ./edge_deployment/
COPY models/ ./models/
COPY utils/ ./utils/

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command: start web dashboard
CMD ["python3", "edge_deployment/raspberry_pi/dashboard.py", "--host", "0.0.0.0", "--port", "8080"]
