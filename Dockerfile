# FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
FROM python:3.9-bullseye
USER root

# 環境変数を定義
# Define env vars
ENV LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja LC_ALL=ja_JP.UTF-8 \
    TZ=JST-9 TERM=xterm PYTHONIOENCODING=utf-8 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /mnt/tomishima2904/word2box

# Define the user and the group
ARG USERNAME=tomishima2904 \
    USER_UID=1098 \
    USER_GID=1000

# docker image に上記変数で定義したユーザーが存在しない場合、ユーザーを登録
# If the user defiend above does not exists, add this
RUN if ! id -u $USERNAME > /dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    build-essential ca-certificates software-properties-common \
    nano vim wget curl file git make xz-utils kmod pipenv locales task-japanese

# 下記からCUDA Toolkitをダウンロード (Debian 11.7, CUDA 11.7)
# https://developer.nvidia.com/cuda-11-7-1-download-archive
RUN wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-debian11-11-7-local_11.7.1-515.65.01-1_amd64.deb && \
    dpkg -i cuda-repo-debian11-11-7-local_11.7.1-515.65.01-1_amd64.deb && \
    cp /var/cuda-repo-debian11-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    add-apt-repository contrib && \
    apt-get update -y && \
    apt-get install -y cuda --no-install-recommends cuda

# 京都大学BARTの環境構築
# Dev env for KU-BART
COPY fairseq ./fairseq/
RUN python -V && cd fairseq && pipenv install

# コンテナ内でルートユーザーとしてのみ振る舞いたいなら以下を消す
# Delete line below if you want to play as root
USER $USERNAME
