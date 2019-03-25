# Start with pytorch base image
FROM pytorch/pytorch

# Install basic utils
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
libglib2.0-0 libxext6 libsm6 libxrender1 \
git mercurial subversion

# nvidia-container-runtime varibles
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

# Install prerequisities vida conda
RUN conda install numpy
RUN conda install pathlib
RUN conda install -c conda-forge visdom 
RUN conda install faiss-gpu cuda91 -c pytorch
RUN conda install -c numba numba
RUN conda install -c soumith torchvision
RUN conda install cudatoolkit
RUN conda install -c numba cudatoolkit=9.1

# Install required python libraries  
RUN apt-get update -yq && apt-get install -yq \
    libhdf5-dev
RUN pip install h5py matplotlib pandas

# Download original code
WORKDIR /
RUN git clone https://github.com/lijx10/SO-Net.git

RUN mv SO-Net sonet

WORKDIR /sonet
RUN mkdir logs	

# Copy changed code
RUN touch models/"__init__.py" && touch data/"__init__.py" && touch util/"__init__.py"
COPY *.sh ./
COPY *.py ./
RUN sh sed_script.sh
RUN mv networks.py ./models && mv *loader* ./data && mv som.py ./util && mv classifier.py ./models
COPY *.ini ./

ENTRYPOINT /bin/bash 