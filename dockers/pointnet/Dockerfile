FROM  tensorflow/tensorflow:latest-gpu

WORKDIR /
RUN apt-get update && apt-get install -y --no-install-recommends \
	libhdf5-dev \
	git \
	wget
	
RUN pip install h5py

RUN git clone https://github.com/charlesq34/pointnet.git

WORKDIR /pointnet

RUN wget 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
RUN unzip modelnet40_ply_hdf5_2048.zip
RUN rm modelnet40_ply_hdf5_2048.zip

ENTRYPOINT /bin/bash 