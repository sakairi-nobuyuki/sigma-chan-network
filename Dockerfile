#FROM nvcr.io/nvidia/pytorch:22.10-py3
FROM nvcr.io/nvidia/pytorch:20.03-py3

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
RUN sudo chown -R $USERNAME .  

RUN pip3 install poetry==${POETRY_VERSION}
#RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python -  && \
RUN poetry config virtualenvs.create false && \
    poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip3 install -r requirements.txt --user --no-deps


### copy python codes
COPY ./sigma_chan_network $HOME/sigma_chan_network/
COPY ./scripts $HOME/scripts/
RUN sudo chown -R $USERNAME .  
