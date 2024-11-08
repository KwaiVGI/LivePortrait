ARG CUDA_VERSION=12.4.1
# **NOTE**: For 4090, change to 12.1.0
#   For H100, change to 12.4.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

LABEL authors="luchentech"

# ====== Install base dependencies ===================
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# install libgl1 for import error described in https://stackoverflow.com/questions/55313610/
#   Alternative solution: pip install opencv-python-headless
RUN apt-get update -y && \
    apt-get install -y python3-pip git curl ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version && python --version

# Install pip s.t. it will be compatible with our PYTHON_VERSION
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

WORKDIR /workspace

RUN python3 -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/KwaiVGI/LivePortrait.git && \
    cd LivePortrait && \
    python3 -m pip install -r requirements.txt

RUN python3 -m pip install --no-cache-dir onnxruntime-gpu==1.17.1 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple