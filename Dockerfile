# Multi-stage production Dockerfile for NeuroScan AI
# Optimized for CPU PyTorch inference with small footprint and non-root security.

# Stage 1: Build & Dependencies
FROM python:3.11-slim as builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime Environment
FROM python:3.11-slim as runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/neuroscan/.local/bin:$PATH \
    FLASK_ENV=production \
    MPLCONFIGDIR=/app/.matplotlib_cache

# Create dedicated non-root application user
RUN groupadd -r neuroscan && useradd -r -g neuroscan -d /home/neuroscan -m neuroscan

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/neuroscan/.local

# Copy application source code
COPY . /app

# Ensure workspace directories exist with correct permissions
RUN mkdir -p /app/instance /app/logs /app/chatbot/static/uploads /app/chatbot/static/reports /app/.matplotlib_cache \
    && chown -R neuroscan:neuroscan /app /home/neuroscan

USER neuroscan

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/login').read()" || exit 1

CMD ["python", "wsgi.py", "--host=0.0.0.0", "--port=5000", "--threads=4"]
