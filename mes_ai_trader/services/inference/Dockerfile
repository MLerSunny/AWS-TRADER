# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY services/inference/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /install /usr/local

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY config.py .
COPY settings.toml .
COPY services/inference/ /app/
COPY utils/ /app/utils/
COPY models/ /app/models/
COPY feature_repo/ /app/feature_repo/

# Create non-root user
RUN useradd -m -r user && chown -R user:user /app
USER user

# Expose port
EXPOSE 8080

# Set Python path
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the server
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"] 