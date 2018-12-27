FROM  tensorflow/tensorflow:latest-gpu

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
	libhdf5-dev \
	git \
	wget

RUN echo "y" | apt-get install g++-6.0
RUN pip install h5py

RUN pip install opencv-python
RUN git clone https://github.com/charlesq34/pointnet2.git

WORKDIR /pointnet2

COPY tf_interpolate_compile.sh tf_ops/3d_interpolation/
RUN cd tf_ops/3d_interpolation/ &&  sh tf_interpolate_compile.sh

COPY tf_sampling_compile.sh tf_ops/sampling
RUN cd tf_ops/sampling/ && sh tf_sampling_compile.sh

COPY tf_grouping_compile.sh tf_ops/grouping/
RUN cd tf_ops/grouping/ && sh tf_grouping_compile.sh

RUN cd data && wget 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
RUN cd data && unzip modelnet40_ply_hdf5_2048.zip
RUN cd data && rm modelnet40_ply_hdf5_2048.zip

COPY *.py ./
#RUN mv ./pointnet2_cls_ssg.py ./models

ENTRYPOINT /bin/bash 
