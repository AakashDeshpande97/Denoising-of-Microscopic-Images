FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart \
    opencv-python-headless scipy bm3d matplotlib numpy

COPY denoising_core.py main.py ./

# Cloud Run injects a PORT environment variable at runtime (usually 8080)
# and expects the container to listen on it - not a hardcoded port.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
