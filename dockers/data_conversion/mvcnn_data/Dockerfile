FROM ubuntu:18.10
MAINTAINER Amit Bakshi <ambakshi@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -yq && apt-get install -yq \
    build-essential \
   	git \
    gcc \
    g++ \
    cmake \
    make \
    libpthread-stubs0-dev
    
RUN git clone --recursive https://github.com/mmp/pbrt-v3/
RUN mv pbrt-v3 app  

WORKDIR /app/build
RUN cmake -G 'Unix Makefiles' ..
RUN /usr/bin/make -j8
RUN ./pbrt_test


RUN apt-get update -yq && apt-get install -yq \
    git \
    apt-utils \
    automake \
    autoconf \
    build-essential \
    python3-dev \
    python3-pip \
    freeglut3-dev \
    autogen \
    libtool \
    python3-numpy
    
RUN apt-get install wget
    
COPY white_background.patch /app/
RUN (cd /app/; git apply white_background.patch)
RUN /usr/bin/make -j8

WORKDIR /app/build
COPY *.pbrt ./
COPY *.py ./
COPY *.ini ./


