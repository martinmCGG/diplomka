# Start with tensorflow base image
FROM tensorflow/tensorflow:1.12.0-gpu

WORKDIR /
# Install some basic utils
RUN apt-get update && apt-get install -y --no-install-recommends \
	libhdf5-dev \
	git \
	wget \
	python-tk

# Install required python libraries    
RUN pip install --upgrade pip
RUN pip install h5py pandas matplotlib

# Download original code
RUN git clone https://github.com/charlesq34/pointnet.git

WORKDIR /pointnet

# Copy changed code
COPY *.py ./
RUN mv pointnet_cls.py ./models
COPY *.ini ./
RUN mkdir logs
