FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04
MAINTAINER caffe-maint@googlegroups.com

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
	
RUN pip install --upgrade pip
RUN pip install requests[security]
RUN pip install scikit-image --ignore-installed six --ignore-installed numpy --ignore-installed scipy
RUN pip install protobuf
RUN pip install opencv-python
RUN pip install futures

ENV CLONE_TAG=master
RUN echo y | apt-get install libhdf5-dev


RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/kanezaki/caffe-rotationnet2.git
COPY Makefile.config caffe-rotationnet2/
COPY *.cpp /opt/caffe/caffe-rotationnet2/src/caffe/layers/
RUN cd caffe-rotationnet2 && make -j 16 && make -j 16 pycaffe

ENV PYCAFFE_ROOT $CAFFE_ROOT/caffe-rotationnet2/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
WORKDIR /
RUN git clone https://github.com/kanezaki/rotationnet.git
WORKDIR /rotationnet

#RUN bash get_modelnet_png.sh 

RUN apt-get update && apt-get install -y --no-install-recommends dtrx libdc1394-22-dev
RUN pip install pandas matplotlib

WORKDIR /opt/caffe/caffe-rotationnet2
RUN /opt/caffe/caffe-rotationnet2/data/ilsvrc12/get_ilsvrc_aux.sh

WORKDIR /rotationnet
RUN mkdir logs
COPY caffe_nets ./caffe_nets

COPY *.prototxt ./Training/
COPY *.py ./
RUN mv classifier.py /opt/caffe/caffe-rotationnet2/python/caffe


