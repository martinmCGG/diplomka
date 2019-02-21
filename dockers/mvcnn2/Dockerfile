FROM pytorch/pytorch

RUN apt-get update -yq && apt-get install -yq \
    libhdf5-dev \
    dtrx \
    wget
    
    
RUN pip install --upgrade pip    
RUN pip install h5py matplotlib pandas

WORKDIR /
RUN git clone https://github.com/jongchyisu/mvcnn_pytorch.git

RUN mv mvcnn_pytorch mvcnn2

#RUN wget http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz
#RUN echo I | dtrx shaded_images.tar.gz && rm shaded_images.tar.gz 

RUN pip install tensorboardX scikit-image

WORKDIR /mvcnn2
#RUN mv /shaded_images/modelnet40_images_new_12x ./modelnet40_images_new_12x
RUN mkdir logs	

COPY *.py ./
RUN mv Trainer.py ./tools/ &&  mv ImgDataset.py ./tools/ && mv Model.py ./models && mv MVCNN.py ./models

ENTRYPOINT /bin/bash 