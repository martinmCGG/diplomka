FROM  tensorflow/tensorflow:latest-gpu

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	git 
	
RUN git clone https://github.com/google/seq2seq.git
WORKDIR /seq2seq

RUN git clone https://github.com/WeiTang114/MVCNN-TensorFlow
RUN mv MVCNN-TensorFlow mvcnn

RUN wget https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar
RUN tar xvf modelnet40v2png_ori4.tar
RUN rm modelnet40v2png_ori4.tar

RUN mkdir data
COPY prepare_image_data.py .
RUN python3 prepare_image_data.py
COPY prepare_data.py .
RUN pip install opencv-python
RUN apt update && apt install -y libsm6 libxext6
RUN echo y | apt-get install libfontconfig1 libxrender1
RUN cd mvcnn && ./prepare_pretrained_alexnet.sh

ENTRYPOINT /bin/bash 