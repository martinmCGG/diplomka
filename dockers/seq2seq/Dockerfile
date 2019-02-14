FROM  tensorflow/tensorflow:latest-gpu

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	git 
	
RUN git clone https://github.com/mingyangShang/SeqViews2SeqLabels && mv SeqViews2SeqLabels seq2seq
WORKDIR /seq2seq

RUN git clone https://github.com/WeiTang114/MVCNN-TensorFlow
RUN mv MVCNN-TensorFlow mvcnn

#RUN wget https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar
#RUN tar xvf modelnet40v2png_ori4.tar
#RUN rm modelnet40v2png_ori4.tar

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN apt update && apt install -y libsm6 libxext6
RUN echo y | apt-get install libfontconfig1 libxrender1
RUN cd mvcnn && ./prepare_pretrained_alexnet.sh

RUN pip install scikit-image

RUN git clone https://github.com/machrisaa/tensorflow-vgg.git
RUN mv tensorflow-vgg vgg
COPY vgg19.npy ./vgg

COPY *.py ./
RUN mv vgg19_trainable.py ./vgg
