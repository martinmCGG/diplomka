# Start with cudnn base image
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04
MAINTAINER caffe-maint@googlegroups.com

# Install basic utilities, python, and python libraries
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
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
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

# Install required python libraries	
RUN pip install --upgrade pip
RUN pip install requests[security]
RUN pip install scikit-image --ignore-installed six --ignore-installed numpy --ignore-installed scipy
RUN pip install protobuf opencv-python futures pandas matplotlib

# Install libhdf
ENV CLONE_TAG=master
RUN echo y | apt-get install libhdf5-dev

# Download custom caffe code and compile it
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/kanezaki/caffe-rotationnet2.git
COPY Makefile.config caffe-rotationnet2/
COPY *.cpp /opt/caffe/caffe-rotationnet2/src/caffe/layers/
RUN cd caffe-rotationnet2 && make -j 16 && make -j 16 pycaffe

# set caffe related environment variables
ENV PYCAFFE_ROOT $CAFFE_ROOT/caffe-rotationnet2/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# Download original network code and compile it
WORKDIR /
RUN git clone https://github.com/kanezaki/rotationnet.git
WORKDIR /rotationnet

# You can download the original image data for modelnet by uncommenting this line
#RUN bash get_modelnet_png.sh 

# Install dtrx - tool for extracting archives
RUN apt-get install -y --no-install-recommends libdc1394-22-dev

# downaload ilsvrc mean file
WORKDIR /opt/caffe/caffe-rotationnet2
RUN /opt/caffe/caffe-rotationnet2/data/ilsvrc12/get_ilsvrc_aux.sh

WORKDIR /rotationnet
RUN mkdir logs
COPY caffe_nets ./caffe_nets

# Copy changed code
COPY *.prototxt ./
COPY *.py ./
RUN mv classifier.py /opt/caffe/caffe-rotationnet2/python/caffe
COPY *.ini ./

