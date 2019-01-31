FROM pytorch/pytorch

WORKDIR /	
RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	git \
	unzip \ 
	libhdf5-dev \
	python-numpy \
	python-scipy
	
RUN git clone https://github.com/kanezaki/pytorch-rotationnet.git
RUN mv pytorch-rotationnet rotationnet
WORKDIR /rotationnet
RUN ls
RUN mkdir logs
RUN bash get_modelnet_png.sh
RUN bash link_images.sh ./modelnet40v1png ./ModelNet40v1 1 

#RUN pip install progressbar
#RUN pip install h5py

COPY *.py ./
ENTRYPOINT /bin/bash 