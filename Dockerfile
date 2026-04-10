FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="6.1"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    poppler-utils \
    tesseract-ocr \
    fonts-dejavu \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN pip install --no-cache-dir -U pip "setuptools<82" wheel packaging

# Pin numpy<2 — Detectron2's C extensions are not NumPy 2.x compatible
RUN pip install --no-cache-dir "numpy<2"

# GPU torch/torchvision
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Detectron2 (from source)
RUN python -m pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git@v0.6"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]