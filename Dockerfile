### Base image
FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04 as base

ENV NV_CUDA_LIB_VERSION "11.4.0-1"

FROM base as base-amd64

ENV NV_CUDA_CUDART_DEV_VERSION 11.4.43-1
ENV NV_NVML_DEV_VERSION 11.4.43-1
ENV NV_LIBCUSPARSE_DEV_VERSION 11.6.0.43-1
ENV NV_LIBNPP_DEV_VERSION 11.4.0.33-1
ENV NV_LIBNPP_DEV_PACKAGE libnpp-dev-11-4=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_VERSION 11.5.2.43-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME libcublas-dev-11-4
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_NVPROF_VERSION 11.4.43-1
ENV NV_NVPROF_DEV_PACKAGE cuda-nvprof-11-4=${NV_NVPROF_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.11.4-1
ENV NCCL_VERSION 2.11.4-1
ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda11.4
FROM base as base-arm64

ENV NV_CUDA_CUDART_DEV_VERSION 11.4.43-1
ENV NV_NVML_DEV_VERSION 11.4.43-1
ENV NV_LIBCUSPARSE_DEV_VERSION 11.6.0.43-1
ENV NV_LIBNPP_DEV_VERSION 11.4.0.33-1
ENV NV_LIBNPP_DEV_PACKAGE libnpp-dev-11-4=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME libcublas-dev-11-4
ENV NV_LIBCUBLAS_DEV_VERSION 11.5.2.43-1
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_NVPROF_VERSION 11.4.43-1
ENV NV_NVPROF_DEV_PACKAGE cuda-nvprof-11-4=${NV_NVPROF_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.11.4-1
ENV NCCL_VERSION 2.11.4-1
ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda11.4


FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtinfo5 libncursesw5 \
    cuda-cudart-dev-11-4=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-11-4=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-11-4=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-11-4=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-11-4=${NV_NVML_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-11-4=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs


####


#FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04 AS base
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04 AS base

FROM base AS base-x86_64

ENV NV_CUDNN_VERSION 8.6.0.163
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"

ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.8"

ENV TARGETARCH x86_64

FROM base-x86_64

ENV TARGETARCH x86_64

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"



RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

### Surface layer image
FROM ubuntu:20.04 AS ubuntu-python

### add a non-root user
RUN apt update && apt install sudo
ARG USERNAME=sigma_chan && GROUPNAME=user && UID=1000 && GID=1000 
ARG PASSWD=$USERNAME && HOME=/home/$USERNAME

RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWD | chpasswd && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME
WORKDIR $HOME

## Configure the environment
ENV POETRY_VERSION=1.2.0a1 \
    POETRY_HOME=$HOME


### install python and poetry
RUN sudo apt update -y && sudo apt install --no-install-recommends -y python3.8 python3-pip python3.8-dev \
    python3-setuptools python3-pip python3-distutils curl \
    build-essential vim curl  && \
    sudo update-alternatives --install /usr/local/bin/python python /usr/bin/python3.8 1 && \
    sudo pip3 install --upgrade pip

ENV PATH $PATH:$HOME/.poetry/bin:$HOME/.local/bin:$HOME/bin:$PATH


### install packages
COPY ./poetry.lock $HOME/
COPY ./pyproject.toml $HOME/
RUN sudo chown -R $USERNAME .  && \
    mkdir -p $HOME/.cache/pip/http && \
    chown -R $USERNAME $HOME/.cache/pip/http

RUN pip3 install poetry==${POETRY_VERSION}
#RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python -  && \
RUN poetry config virtualenvs.create false && \
    poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip3 install -r requirements.txt --user --no-deps

#ADD ./sigma_chan_network $HOME/sigma_chan_network
#ENV PYTHONPATH $PYTHONPATH:$HOME/sigma_chan_network
#:/usr/lib/python38.zip:/usr/lib/python3.8/lib-dynload:$HOME/.local/lib/python3.8/site-packages:/usr/local/lib/python3.8/dist-packages:/usr/lib/python3/dist-packages

FROM ubuntu-python AS ubuntu-python-surface

ARG USERNAME=sigma_chan && GROUPNAME=user && UID=1000 && GID=1000 
ARG PASSWD=$USERNAME && HOME=/home/$USERNAME

USER $USERNAME
WORKDIR $HOME

### copy python codes
COPY ./sigma_chan_network $HOME/sigma_chan_network/
COPY ./scripts $HOME/scripts/
COPY ./main.py $HOME/
RUN sudo chown -R $USERNAME .
