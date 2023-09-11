FROM kunlp/jumanpp
USER root

# 環境変数を定義
# Define env vars
ENV LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja LC_ALL=ja_JP.UTF-8 \
    TZ=JST-9 TERM=xterm PYTHONIOENCODING=utf-8

WORKDIR /work/tomishima2904/explore_conceptnet

# 上から, python関連, build関連, その他パッケージ, rust, pipのupgrade, cargoへのパス通し
RUN apk update && apk add \
    python3 python3-dev py3-pip \
    alpine-sdk openssl-dev make cmake libgomp \
    nano vim wget curl file git shadow \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && pip install --upgrade pip \
    && echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> $HOME/.profile

# ~/.cargo/bin へのパスを通して cargo・rustc・rustup等を使えるようにする
# 上記パッケージをインストールしないとtransformersが pip install できないため
ENV PATH="/root/.cargo/bin:$PATH"

COPY requirements.txt ./

# 下記リンクを参考にpytorchをインストール
# https://pytorch.org/get-started/locally/
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

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

# コンテナ内でルートユーザーとしてのみ振る舞いたいなら以下を消す
# Delete line below if you want to play as root
USER $USERNAME
