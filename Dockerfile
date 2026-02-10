# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY . .

# Create data directory for snapshots and set ownership
RUN mkdir -p /app/data/screener_snapshots \
    && groupadd -r appgroup && useradd -r -g appgroup -s /bin/bash appuser \
    && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Default command: Live paper trading (overridable)
CMD ["python", "-m", "src.main", "--mode", "paper"]
