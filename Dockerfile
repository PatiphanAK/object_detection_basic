# Use Python 3.12 slim
FROM python:3.12-slim-trixie

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies with uv (much faster!)
RUN uv sync --frozen --no-dev

# Copy application code
COPY main.py .
COPY model_service.py .

# Copy model weights (adjust path as needed)
COPY ./weights/best/best.pt .

# Create necessary directories
RUN mkdir -p uploads results gradcam_results

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
