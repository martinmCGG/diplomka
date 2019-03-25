# Start with tensorflow base image
FROM  tensorflow/tensorflow:1.12.0-gpu

WORKDIR /

# Install some basic utils
RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	git 

# Install required python libraries  
RUN pip install --upgrade pip
RUN pip install scikit-image

# Download original code
RUN git clone https://github.com/mingyangShang/SeqViews2SeqLabels && mv SeqViews2SeqLabels seq2seq
WORKDIR /seq2seq

# Copy changed code
COPY *.py ./
COPY *.ini ./
