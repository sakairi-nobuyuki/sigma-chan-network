FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04 as base

FROM base as base-x86_64

ENV NV_CUDNN_VERSION 8.6.0.163
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"

ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.8"

ENV TARGETARCH x86_64

FROM base as base-arm64

ENV NV_CUDNN_VERSION 8.6.0.163
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"

ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.8"

FROM base-x86_64

ENV TARGETARCH x86_64

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*



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
RUN sudo apt install --no-install-recommends -y python3.8 python3-pip python3.8-dev \
    python3-setuptools python3-pip python3-distutils curl \
    build-essential vim curl  && \
    sudo update-alternatives --install /usr/local/bin/python python /usr/bin/python3.8 1 && \
    sudo pip3 install --upgrade pip

ENV PATH $PATH:$HOME/.poetry/bin:$HOME/.local/bin:$HOME/bin:$PATH


### install packages
COPY ./poetry.lock $HOME/
COPY ./pyproject.toml $HOME/
RUN sudo chown -R $USERNAME .  && mkdir -p $HOME/.cache/pip/http && chown -R $USERNAME $HOME/.cache/pip/http

RUN pip3 install poetry==${POETRY_VERSION}
#RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python -  && \
RUN poetry config virtualenvs.create false && \
    poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip3 install -r requirements.txt --user --no-deps


### copy python codes
COPY ./sigma_chan_network $HOME/sigma_chan_network/
COPY ./scripts $HOME/scripts/
RUN sudo chown -R $USERNAME .  
