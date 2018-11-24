FROM ubuntu:bionic
MAINTAINER Anastasios Zouzias <my_last_name@gmail.com>
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/include/

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libtool \
        libatlas-base-dev \
        libboost-all-dev \
        libboost-python-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        libglu1-mesa-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        autotools-dev \
        dtrx \
        automake \
        autoconf \
        cmake \
        xorg-dev 	\
        flex \
        bison \
    python-h5py \
	python-numpy \
	python-scipy \
	python-setuptools \
	python-pip  &&\
    	rm -rf /var/lib/apt/lists/*

RUN apt-get -qq update && apt-get install -q -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt-get -qq update && apt-get install -qy g++ gcc git wget

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install futures

ENTRYPOINT /bin/bash 