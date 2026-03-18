# Cerberus Trading System
FROM python:3.12-slim-bookworm

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install empire-core shared library (copied from monorepo as _empire_core/)
COPY _empire_core/ /tmp/empire-core/
RUN pip install --no-cache-dir /tmp/empire-core/ && rm -rf /tmp/empire-core/

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY . .

# Clean up build artifacts that shouldn't be in the image
RUN rm -rf /app/_empire_core

# Create data directory for snapshots and set ownership
RUN mkdir -p /app/data/screener_snapshots \
    && groupadd -r appgroup && useradd -r -g appgroup -s /bin/bash appuser \
    && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Default command: Live paper trading (overridable)
CMD ["python", "-m", "src.main", "--mode", "paper"]
