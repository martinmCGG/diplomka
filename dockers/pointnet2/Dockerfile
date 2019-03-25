# Start with tensorflow base image
FROM  tensorflow/tensorflow:1.12.0-gpu

WORKDIR /
# Install some basic utils
RUN apt-get update && apt-get install -y --no-install-recommends \
	libhdf5-dev \
	git \
	wget

# Install g++ compiler
RUN echo "y" | apt-get install g++-6.0

# Install required python libraries 
RUN pip install h5py opencv-python

# Download original code
RUN git clone https://github.com/charlesq34/pointnet2.git

WORKDIR /pointnet2

# Compile custom tensorflow operations
COPY tf_interpolate_compile.sh tf_ops/3d_interpolation/
RUN cd tf_ops/3d_interpolation/ &&  sh tf_interpolate_compile.sh

COPY tf_sampling_compile.sh tf_ops/sampling
RUN cd tf_ops/sampling/ && sh tf_sampling_compile.sh

COPY tf_grouping_compile.sh tf_ops/grouping/
RUN cd tf_ops/grouping/ && sh tf_grouping_compile.sh

# Copy changed code
RUN mkdir logs
COPY *.py ./
RUN mv pointnet2_cls_ssg.py ./models
COPY *.ini ./

ENTRYPOINT /bin/bash 

