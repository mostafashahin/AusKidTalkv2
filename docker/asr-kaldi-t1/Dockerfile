FROM ubuntu
ENV TZ=Australia/Sydney

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    python3 \
    python3-pip \
    curl \
    gnupg \
    lsb-release \
    ffmpeg

RUN git clone https://github.com/mostafashahin/AusKidTalkv2.git /opt/AusKidTalkv2

RUN pip3 install numpy==1.21 pandas pydub ibm_watson numba librosa

RUN pip3 install --upgrade google google-cloud google-cloud-speech google-cloud-storage

RUN pip3 install boto3

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    docker-ce docker-ce-cli containerd.io

COPY run.sh /opt/AusKidTalkv2

WORKDIR /opt/AusKidTalkv2
