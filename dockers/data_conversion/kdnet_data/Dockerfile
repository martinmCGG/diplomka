# Start with ubuntu base image
FROM ubuntu:bionic

# Install basic utils and python
WORKDIR /
RUN apt-get update -yq && apt-get install -yq \
    apt-utils \
    build-essential \
    python3-dev \
    python3-pip \
    python3-numpy \
    libhdf5-dev \
    git

# Install python libraries    
RUN pip3 install h5py pandas pathlib

# Download original kdnet code
RUN git clone https://github.com/Regenerator/kdnets.git

# Copy changed files
WORKDIR kdnets
RUN mkdir logs
COPY *.py ./
RUN mv modelnet40.py ./lib/processors && mv shapenetcore.py ./lib/processors
COPY *.ini ./

ENTRYPOINT /bin/bash 
