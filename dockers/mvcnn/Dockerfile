# Start with tensorflow base image
FROM  tensorflow/tensorflow:1.12.0-gpu
WORKDIR /

# Install some basic utils
RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	git \
	libglib2.0-0

# Install required python libraries
RUN pip install --upgrade pip
RUN pip install hickle scikit-learn opencv-python

# Download original code
RUN git clone https://github.com/WeiTang114/MVCNN-TensorFlow.git
RUN mv MVCNN-TensorFlow mvcnn
WORKDIR /mvcnn 

# Install required libraries
RUN apt update && apt install -y libsm6 libxext6
RUN echo y | apt-get install libfontconfig1 libxrender1

# Prepare pretrained 2D convolutional network
RUN ./prepare_pretrained_alexnet.sh

# Copy changed code
COPY *.py ./
COPY *.ini ./

ENTRYPOINT /bin/bash 
