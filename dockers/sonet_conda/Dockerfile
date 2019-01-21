#FROM continuumio/miniconda3
#RUN conda create -n env python=3.6
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
libglib2.0-0 libxext6 libsm6 libxrender1 \
git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda2-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
/bin/bash ~/anaconda.sh -b -p /opt/conda && \
rm ~/anaconda.sh && \
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
dpkg -i tini.deb && \
rm tini.deb && \
apt-get clean

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

RUN conda install pathlib
RUN conda install -c conda-forge visdom 
RUN conda install faiss-gpu cuda90 -c pytorch 
RUN conda install numpy
RUN conda install cudatoolkit=9.0
RUN conda install -c numba numba
RUN conda install pytorch=0.4.1 cuda90 -c pytorch
RUN conda install -c soumith torchvision


RUN git clone https://github.com/lijx10/SO-Net.git

RUN mv SO-Net sonet

WORKDIR /sonet
RUN mkdir logs

RUN touch models/"__init__.py" && touch data/"__init__.py" && touch util/"__init__.py"
COPY *.sh ./
COPY *.py ./
COPY sonet_data ./sonet_data
RUN sh sed_script.sh
RUN mv networks.py ./models && mv *loader* ./data && mv som.py ./util


#ENTRYPOINT /bin/bash 
