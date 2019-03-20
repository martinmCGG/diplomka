FROM ubuntu:bionic

WORKDIR /
RUN apt-get update -yq && apt-get install -yq \
    apt-utils \
    build-essential \
    python3-dev \
    python3-pip \
    python3-numpy \
    libhdf5-dev \
    git
    
RUN pip3 install h5py pandas pathlib

RUN git clone https://github.com/Regenerator/kdnets.git

WORKDIR kdnets
RUN mkdir logs
COPY *.py ./
RUN mv modelnet40.py ./lib/processors && mv shapenetcore.py ./lib/processors
COPY *.ini ./

ENTRYPOINT /bin/bash 
