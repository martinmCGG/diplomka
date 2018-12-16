FROM ubuntu:bionic

WORKDIR /
RUN apt-get update -yq && apt-get install -yq \
    apt-utils \
    build-essential \
    python3-dev \
    python3-pip \
    python3-numpy \
    libhdf5-dev
    
RUN pip3 install h5py
    
COPY *.py /