FROM python:3.12-slim

WORKDIR /usr/src/app

# System deps for Pillow / torch CPU wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU torch first (smaller than default CUDA wheels), then the rest.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py predictor.py .

ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=5050
ENV MODEL_CACHE_DIR=/usr/src/app/models
ENV HF_HOME=/usr/src/app/models/hf
ENV TRANSFORMERS_CACHE=/usr/src/app/models/hf
ENV DEVICE=cpu

EXPOSE 5050

CMD ["python3", "main.py"]
