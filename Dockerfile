# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies (needed for compiling some ML packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ /app/src/
COPY app/ /app/app/
COPY params.yaml .
# Environment variables are passed at runtime, so we don't copy .env here

# Copy pre-trained artifacts
# Make sure you've run the DVC/Training pipeline before building!
COPY artifacts/ /app/artifacts/

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI application via uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
