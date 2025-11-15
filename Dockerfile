# Use Python 3.12 slim (Bookworm is more stable than Trixie)
FROM python:3.12-slim-bookworm

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy dependency files first (for better caching)
COPY pyproject.toml uv.lock ./

# Install dependencies with uv (much faster!)
RUN uv sync --frozen --no-dev

# Copy model weights (separate layer for easier updates)
COPY weights/best/best.pt /app/best.pt

# Copy application code (changes frequently, so put last)
COPY main.py model_service.py ./

# Create necessary directories
RUN mkdir -p uploads results gradcam_results

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application using uv
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
