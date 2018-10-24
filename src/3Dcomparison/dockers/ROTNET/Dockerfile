FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04
MAINTAINER caffe-maint@googlegroups.com

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
	python-scipy \
	python-setuptools \
	python-pip  &&\
    	rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT
RUN pip install --upgrade pip
RUN pip install scikit-image --ignore-installed six --ignore-installed numpy --ignore-installed scipy

# FIXME: clone a specific git tag and use ARG instead of ENV once DockerHub supports this.
ENV CLONE_TAG=master
RUN apt-get install libhdf5-dev
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/Alobal123/test2.git
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/kanezaki/caffe-rotationnet2

RUN cp test2/Makefile.config caffe-rotationnet2/Makefile.config
RUN cd caffe-rotationnet2 && make && make pycaffe



ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig



