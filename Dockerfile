FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Install curl for health check
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables for file persistence
ENV PERSISTENCE_TYPE=file
ENV DATA_DIRECTORY=/app/data
# Note: Not hardcoding the COHERE_API_KEY here, it should be set via environment variable when running container
# Create data directory with proper permissions
RUN mkdir -p /app/data && chmod 755 /app/data

# Run the main application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
