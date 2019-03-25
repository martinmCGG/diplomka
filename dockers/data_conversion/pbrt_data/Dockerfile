# Start with ubuntu base image
FROM ubuntu:18.10
MAINTAINER Amit Bakshi <ambakshi@gmail.com>

# Install basic utils and python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -yq && apt-get install -yq \
    build-essential \
   	git \
    gcc \
    g++ \
    cmake \
    make \
    libpthread-stubs0-dev \
    automake \
    autoconf \
    python3-dev \
    python3-pip \
    freeglut3-dev \
    autogen \
    libtool \
    python3-numpy \
    wget
   
#Download pbrt
RUN git clone --recursive https://github.com/mmp/pbrt-v3/
RUN mv pbrt-v3 app  

#Build pbrt
WORKDIR /app/build
RUN cmake -G 'Unix Makefiles' ..
RUN /usr/bin/make -j8
RUN ./pbrt_test

# Apply patch to have white background
COPY white_background.patch /app/
RUN (cd /app/; git apply white_background.patch)
RUN /usr/bin/make -j8

# Copy changed filess
WORKDIR /app/build
COPY *.pbrt ./
COPY *.py ./
COPY *.ini ./


