FROM tensorflow/tensorflow:1.12.0-gpu

WORKDIR /
RUN apt-get update && apt-get install -y --no-install-recommends \
	libhdf5-dev \
	git \
	wget \
	python-tk

RUN pip install --upgrade pip
RUN pip install h5py pandas matplotlib

RUN git clone https://github.com/charlesq34/pointnet.git

WORKDIR /pointnet

RUN mkdir data
#RUN cd data && wget 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#RUN cd data && unzip modelnet40_ply_hdf5_2048.zip
#RUN cd data && rm modelnet40_ply_hdf5_2048.zip

COPY *.py ./
RUN mv pointnet_cls.py ./models
COPY *.ini ./
RUN mkdir logs
