FROM  tensorflow/tensorflow:1.12.0-gpu

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	git 
	

RUN pip install --upgrade pip
RUN pip install scikit-image

RUN git clone https://github.com/machrisaa/tensorflow-vgg.git
RUN mv tensorflow-vgg vgg
WORKDIR vgg
COPY vgg19.npy ./
COPY *.py ./
COPY *.ini ./
