FROM pbrt
RUN apt-get update -yq && apt-get install -yq \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-numpy
    
COPY white_background.patch /app/
RUN (cd /app/; git apply white_background.patch)
RUN /usr/bin/make -j8
COPY *.py /app/build/

WORKDIR /app/build