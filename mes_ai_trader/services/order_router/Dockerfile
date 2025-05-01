# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY services/order_router/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final image
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -r user

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY config.py .
COPY settings.toml .
COPY services/order_router/ /app/
COPY utils/ /app/utils/

# Set ownership
RUN chown -R user:user /app

# Switch to non-root user
USER user

# Expose port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["uvicorn", "router:app", "--host", "0.0.0.0", "--port", "8000"] 