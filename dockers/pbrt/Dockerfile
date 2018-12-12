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


RUN echo y | apt-get install software-properties-common
RUN wget http://www.antiprism.com/files/antiprism-0.24.1.tar.gz
RUN tar zxf antiprism-0.24.1.tar.gz
RUN rm antiprism-0.24.1.tar.gz
RUN cd antiprism-0.24.1 && ls && autoconf configure.ac && ./configure
RUN echo "reset here "
RUN cd antiprism-0.24.1 && make -j 8
#RUN cd antiprism-0.24.1 && make install-strip -j 8 .

COPY *.py /app/build/
COPY *.pbrt /app/build/

WORKDIR /app/build
