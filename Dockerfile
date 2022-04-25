FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

#### System package (uses default Python 3 version in Ubuntu 20.04)
RUN apt-get update -y && \
    apt-get install -y \
        git python3 python3-dev libpython3-dev python3-pip sudo pdsh \
        htop llvm-9-dev tmux zstd software-properties-common build-essential autotools-dev \
        nfs-common pdsh cmake g++ gcc curl wget vim less unzip htop iftop iotop ca-certificates ssh \
        rsync iputils-ping net-tools libcupti-dev libmlx4-1 infiniband-diags ibutils ibverbs-utils \
        rdmacm-utils perftest rdma-core nano && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --upgrade pip && \
    pip install gpustat

COPY requirements.txt .
RUN pip install -r requirements.txt && pip cache purge

RUN mkdir -p /home/block-recurrent-transformer
COPY . /home/block-recurrent-transformer
WORKDIR /home/block-recurrent-transformer

