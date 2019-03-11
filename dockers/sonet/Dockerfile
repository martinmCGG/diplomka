FROM pytorch/pytorch

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
libglib2.0-0 libxext6 libsm6 libxrender1 \
git mercurial subversion

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

RUN conda install numpy
RUN conda install pathlib
RUN conda install -c conda-forge visdom 
RUN conda install faiss-gpu cuda91 -c pytorch
RUN conda install -c numba numba
RUN conda install -c soumith torchvision
RUN conda install cudatoolkit
RUN conda install -c numba cudatoolkit=9.1


RUN apt-get update -yq && apt-get install -yq \
    libhdf5-dev
RUN pip install h5py matplotlib pandas

WORKDIR /
RUN git clone https://github.com/lijx10/SO-Net.git

RUN mv SO-Net sonet

WORKDIR /sonet
RUN ls -l
RUN mkdir logs	

RUN touch models/"__init__.py" && touch data/"__init__.py" && touch util/"__init__.py"
COPY *.sh ./
COPY *.py ./
COPY sonet_data ./sonet_data
RUN sh sed_script.sh
RUN mv networks.py ./models && mv *loader* ./data && mv som.py ./util && mv classifier.py ./models
COPY *.ini ./

ENTRYPOINT /bin/bash 