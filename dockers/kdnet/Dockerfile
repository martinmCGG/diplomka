FROM pytorch/pytorch

WORKDIR /	
RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	git \
	unzip \ 
	libhdf5-dev \
	python-numpy \
	python-scipy
	
RUN git clone https://github.com/fxia22/kdnet.pytorch
RUN mv kdnet.pytorch kdnet
WORKDIR /kdnet

RUN mkdir data

RUN cd data && wget 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
RUN cd data && unzip modelnet40_ply_hdf5_2048.zip
RUN cd data && rm modelnet40_ply_hdf5_2048.zip

RUN pip install progressbar
RUN pip install h5py

COPY *.py ./
ENTRYPOINT /bin/bash 