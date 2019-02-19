FROM  tensorflow/tensorflow:1.12.0-gpu
WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	git \
	libglib2.0-0

RUN git clone https://github.com/WeiTang114/MVCNN-TensorFlow.git

RUN mv MVCNN-TensorFlow mvcnn
WORKDIR /mvcnn 

#RUN wget https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar
#RUN tar xvf modelnet40v2png_ori4.tar
#RUN rm modelnet40v2png_ori4.tar

RUN mkdir data
#COPY prepare_image_data.py .
#RUN python3 prepare_image_data.py

RUN pip install --upgrade pip
RUN pip install hickle scikit-learn
RUN pip install opencv-python
RUN apt update && apt install -y libsm6 libxext6
RUN echo y | apt-get install libfontconfig1 libxrender1
RUN ./prepare_pretrained_alexnet.sh

COPY *.py ./

ENTRYPOINT /bin/bash 
