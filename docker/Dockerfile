FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04
# FROM nvidia/cuda:12.5.0-devel-ubuntu22.04
# FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /LivePortrait

# # 必要なシステムパッケージをインストールする
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libx11-dev

# システムレベルで必要なパッケージをインストールします
RUN apt-get update && apt-get install -y python3-pip python3-dev

# pip をアップグレードする
RUN pip install --upgrade pip

# 必要な Python パッケージをインストールする
COPY requirements.txt .
RUN pip install -r requirements.txt

# タイムゾーンを設定する
# 例としてAsia/Tokyoを設定
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
# タイムゾーンを設定
ENV TZ Asia/Tokyo

RUN apt-get install libopencv-dev -y

# 作業ディレクトリを設定する
WORKDIR /LivePortrait
