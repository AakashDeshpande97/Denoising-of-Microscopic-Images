FROM python:3.11-slim

WORKDIR /app

# opencv/scipy/bm3d need these system libs to build/import cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (much smaller/faster than the default GPU build).
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

RUN pip install --no-cache-dir \
    opencv-python-headless scipy bm3d matplotlib numpy

COPY compare_denoising.py .

ENTRYPOINT ["python", "compare_denoising.py"]