FROM ubuntu

RUN apt-get -y update
RUN apt-get install -y python3 python3-pip

RUN pip install numba pandas

COPY script/scoreAnn.py  /opt/script/
COPY scoreit.sh /opt/
RUN mkdir -p /opt/script/tools

ADD txtgrid_master /opt/script/tools/txtgrid_master

WORKDIR /opt/
