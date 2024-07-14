FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04
# FROM nvidia/cuda:12.5.0-devel-ubuntu22.04
# FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /LivePortrait

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libx11-dev

# Install required system-level packages
RUN apt-get update && apt-get install -y python3-pip python3-dev

# Upgrade pip
RUN pip install --upgrade pip

# Install required Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set the timezone
# Example: Set to Asia/Tokyo
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
# Set timezone
ENV TZ Asia/Tokyo

RUN apt-get install libopencv-dev -y

# Set the working directory
WORKDIR /LivePortrait
