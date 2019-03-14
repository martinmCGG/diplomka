FROM ubuntu:bionic

WORKDIR /
RUN apt-get update -yq && apt-get install -yq \
    apt-utils \
    build-essential \
    python-dev \
    python-pip \
    python-numpy \
    libhdf5-dev \
    git
    
RUN pip install h5py pandas pathlib

RUN git clone https://github.com/Regenerator/kdnets.git

WORKDIR kdnets
RUN mkdir logs
COPY *.py ./
RUN mv modelnet40.py ./lib/processors && mv shapenet.py ./lib/processors
COPY *.ini ./

ENTRYPOINT /bin/bash 
