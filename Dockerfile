FROM kunlp/jumanpp
USER root

# 環境変数を定義
# Define env vars
ENV LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja LC_ALL=ja_JP.UTF-8 \
    TZ=JST-9 TERM=xterm PYTHONIOENCODING=utf-8 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /work/tomishima2904/explore_conceptnet

RUN apk update && apk add \
    python3 python3-dev py3-pip alpine-sdk nano vim wget curl file git make
