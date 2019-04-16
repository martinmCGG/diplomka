# Start with cudnn base image
#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Install basic utilities, python, and python libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libcgal-dev \
        libeigen3-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-setuptools \
        python-scipy \
        rsync \
        software-properties-common \
        vim \
        zip && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
        rm get-pip.py

# Install scikit and cython
RUN pip --no-cache-dir install \
        scikit-build \
        Cython

# Set environment variables
ENV WORKSPACE=/workspace
ENV PATH="${WORKSPACE}/bin:${PATH}"
ARG WORKSPACE_BIN=$WORKSPACE/bin
WORKDIR $WORKSPACE_BIN
ARG OCNN_ROOT=$WORKSPACE/ocnn
WORKDIR $OCNN_ROOT

# Download original code
ARG OCNN_COMMIT=origin
RUN git clone https://github.com/Microsoft/O-CNN.git . && git reset --hard $OCNN_COMMIT

# Build network
RUN cd ocnn/octree && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DOUTPUT_DIRECTORY=$WORKSPACE_BIN .. && \
    make && cd ../ && rm -rf build

# Install libraries for multigpu training
ARG NCCL_COMMIT=286916a1a37ca1fe8cd43e280f5c42ec29569fc5
WORKDIR $WORKSPACE
RUN git clone https://github.com/NVIDIA/nccl.git && cd nccl && git reset --hard $NCCL_COMMIT && \
    make -j install && cd .. && rm -rf nccl

# Install caffe
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT
ARG CAFFE_COMMIT=6bfc5ca8f7c2a4b7de09dfe7a01cf9d3470d22b3
RUN git clone https://github.com/BVLC/caffe.git . && git reset --hard $CAFFE_COMMIT && \
    rsync -a $OCNN_ROOT/caffe/ ./ && ex -sc '83i|set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")' -cx CMakeLists.txt && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd ..
RUN mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
    make -j"$(nproc)" && \
    mkdir $CAFFE_ROOT/include/caffe/proto && \
    cp $CAFFE_ROOT/build/include/caffe/proto/caffe.pb.h $CAFFE_ROOT/include/caffe/proto/caffe.pb.h

# Prepare netwrok for use 
WORKDIR $OCNN_ROOT
RUN cd ocnn/octree && pip install . && cd ../caffe && pip install .

ARG OCNN_TOOLS_ROOT=$WORKSPACE/ocnn_tools
WORKDIR $OCNN_TOOLS_ROOT

ARG OCNN_TOOLS_COMMIT=origin
RUN git clone https://github.com/wang-ps/O-CNN.git . && git reset --hard $OCNN_TOOLS_COMMIT
RUN cd virtual_scanner && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make && \
    cp virtualscanner $WORKSPACE_BIN && cd ../ && rm -rf build && \
    pip install .

# set caffe related environment variables
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# Install additional python librarires
RUN pip install --upgrade pip    
RUN pip install pathlib matplotlib pandas lmdb

# Copy changed code
WORKDIR /workspace
RUN mkdir logs
COPY examples ./
COPY *.py ./
COPY *.ini ./

