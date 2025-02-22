FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (including libgomp1)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libgl1-mesa-glx \  
    git \
    python3-dev \
    libopenblas-dev \
    libomp-dev \      
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install python-multipart
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app
COPY ./data /data

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
