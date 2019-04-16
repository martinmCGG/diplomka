# Start with pytorch base image
FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

# Install some basic utils
RUN apt-get update -yq && apt-get install -yq \
    libhdf5-dev \
    dtrx \
    wget
    
# Install required python libraries    
RUN pip install --upgrade pip    
RUN pip install h5py matplotlib pandas scikit-image

WORKDIR /
# Download original code
RUN git clone https://github.com/jongchyisu/mvcnn_pytorch.git
RUN mv mvcnn_pytorch mvcnn2

WORKDIR /mvcnn2
RUN mkdir logs	

#RUN pip install torchvision

# Copy changed code
COPY *.py ./
RUN mv Trainer.py ./tools/ &&  mv ImgDataset.py ./tools/ && mv Model.py ./models && mv MVCNN.py ./models
COPY *.ini ./

ENTRYPOINT /bin/bash 
