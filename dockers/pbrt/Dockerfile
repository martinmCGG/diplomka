FROM pbrt
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

COPY *.py /app/build/
COPY *.pbrt /app/build/

WORKDIR /app/build
