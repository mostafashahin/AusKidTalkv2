FROM ubuntu

RUN apt-get update

RUN apt-get -y install git python3.7 sox python3-venv python3-pip

RUN pip3 install -U pip

#git the code repository
RUN git clone https://github.com/mostafashahin/AusKidTalkv2.git /opt/AusKidTalkv2/ && \
    cd /opt/AusKidTalkv2 && \
    pip3 install -r requirements.txt

#git the pyAudioAnalysis tool in tools directory

COPY run_aligner.sh /opt/AusKidTalkv2

RUN chmod 777 /opt/AusKidTalkv2/run_aligner.sh

CMD ./opt/AusKidTalk/run_aligner.sh
