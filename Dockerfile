FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements files
COPY requirements*.txt ./

# Install in correct order to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir demucs

# Copy the script
COPY auto_chart.py .

# Create directories
RUN mkdir -p /input /output

ENTRYPOINT ["python", "auto_chart.py"]