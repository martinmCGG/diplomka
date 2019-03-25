# Start with ubuntu base image
FROM ubuntu:bionic
MAINTAINER Anastasios Zouzias <my_last_name@gmail.com>
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/include/

# Install all prerequisities, basic utils and python
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

# Install additional python libraries
RUN pip install --upgrade pip
RUN pip install opencv-python futures pathlib

ENV CLONE_TAG=master

# Download and install zlib
RUN wget http://zlib.net/zlib-1.2.11.tar.gz
RUN tar zxf zlib-1.2.11.tar.gz
RUN rm zlib-1.2.11.tar.gz
RUN cd zlib-1.2.11 && ./configure && make -j 8 && make install

# Download and install tbb
RUN wget https://github.com/01org/tbb/releases/download/2018_U6/tbb2018_20180822oss_lin.tgz
RUN dtrx tbb2018_20180822oss_lin.tgz
RUN rm tbb2018_20180822oss_lin.tgz

# Download and install ilmbase
RUN wget https://github.com/openexr/openexr/releases/download/v2.3.0/ilmbase-2.3.0.tar.gz
RUN tar zxf ilmbase-2.3.0.tar.gz
RUN rm ilmbase-2.3.0.tar.gz
RUN cd ilmbase-2.3.0 && autoconf && ./configure --prefix="/usr/include/openexr" && make -j 32 && make install

# Download and install openexr
RUN wget https://github.com/openexr/openexr/releases/download/v2.3.0/openexr-2.3.0.tar.gz
RUN tar zxf openexr-2.3.0.tar.gz
RUN rm openexr-2.3.0.tar.gz
RUN cd openexr-2.3.0 && autoconf && ./configure --prefix="/usr/include/openexr" --with-ilmbase-prefix="/usr/include/openexr"  && make -j 32 && make install

# Donwload and install jemalloc
RUN wget https://github.com/jemalloc/jemalloc/releases/download/4.0.3/jemalloc-4.0.3.tar.bz2
RUN dtrx jemalloc-4.0.3.tar.bz2 && mv jemalloc-4.0.3 jemalloc
RUN rm jemalloc-4.0.3.tar.bz2
RUN cd jemalloc && autoconf && ./configure && make -j 8 && make install

# Download and install cppunit
RUN wget http://dev-www.libreoffice.org/src/cppunit-1.14.0.tar.gz
RUN dtrx cppunit-1.14.0.tar.gz && mv cppunit-1.14.0 cppunit
RUN rm cppunit-1.14.0.tar.gz
RUN cd cppunit && ./configure && make -j 8 && make install

# Download nad install glfw
RUN wget https://github.com/glfw/glfw/releases/download/3.2.1/glfw-3.2.1.zip
RUN dtrx glfw-3.2.1.zip
RUN rm glfw-3.2.1.zip
RUN cd glfw-3.2.1 && cmake -DBUILD_SHARED_LIBS=ON . && make -f Makefile -j 8 && make -f Makefile install

#  Download and install openvdb
RUN wget https://github.com/AcademySoftwareFoundation/openvdb/archive/v5.2.0.zip
RUN echo "H" | dtrx v5.2.0.zip 
RUN mv openvdb-5.2.0 openvdb

# Copy Makefile with correctly set paths and variables
COPY Makefile openvdb/openvdb/Makefile

# Make some neccessery links
RUN ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py27.a /usr/lib/x86_64-linux-gnu/libboost_python2.7.a
RUN ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py27.so /usr/lib/x86_64-linux-gnu/libboost_python2.7.so

# Build openvdb
RUN cd openvdb/openvdb && make -j 8 . && make install -j 8 .

# Set environment variables
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH":/usr/local/openvdb/lib/:/usr/include/ilmbase-2.3.0/Half/.libs:/usr/local/lib/"
ENV PYTHONPATH=$PYTHONPATH:/usr/local/openvdb/python/lib/python2.7/

# Copy changed files

WORKDIR /
COPY *.py ./
COPY *.ini ./
