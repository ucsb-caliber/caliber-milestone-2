FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    fonts-dejavu \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin setuptools to keep pkg_resources (setuptools>=82 removed it)
# Also install packaging explicitly (commonly needed during builds)
RUN pip install --no-cache-dir -U pip "setuptools<82" wheel packaging

# CPU torch/torchvision
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.1.2 torchvision==0.16.2

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Re-assert pin in case requirements.txt upgraded setuptools
RUN python -m pip install --no-cache-dir -U pip "setuptools<82" wheel packaging

# Detectron2 (from source)
RUN python -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git@v0.6"

COPY . /app

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]