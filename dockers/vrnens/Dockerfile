# Start with cuDNN base image
FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04
MAINTAINER Kai Arulkumaran <design@kaixhin.com>

# Install git, wget, python-dev, pip, BLAS + LAPACK and other dependencies
RUN apt-get update && apt-get install -y \
  gfortran \
  git \
  wget \
  liblapack-dev \
  libopenblas-dev \
  python-dev \
  python-pip \
  python-nose
  
  
# Remove OS-installed six
RUN rm /usr/lib/python2.7/dist-packages/six*
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install scipy
RUN pip install matplotlib
RUN pip install pandas



# Set CUDA_ROOT
ENV CUDA_ROOT /usr/local/cuda/bin

# Install CMake 3
RUN cd /root && wget http://www.cmake.org/files/v3.8/cmake-3.8.1.tar.gz && \
  tar -xf cmake-3.8.1.tar.gz && cd cmake-3.8.1 && \
  ./configure && \
  make -j "$(nproc)" && \
  make install

# Install Cython
RUN pip install Cython

# Clone libgpuarray repo and move into it
RUN cd /root && git clone https://github.com/Theano/libgpuarray.git && cd libgpuarray && \
# Make and move into build directory
  mkdir Build && cd Build && \
# CMake
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr && \
# Make
  make -j"$(nproc)" && \
  make install
# Install pygpu
RUN cd /root/libgpuarray && \
  python setup.py build_ext -L /usr/lib -I /usr/include && \
  python setup.py install

# Install bleeding-edge Theano
RUN pip install --upgrade pip
RUN pip install --upgrade six
RUN pip install path.py
RUN pip install Theano
RUN pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
RUN pip install Lasagne
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
# Set up .theanorc for CUDA
RUN echo "[global]\ndevice=cuda\nfloatX=float32\noptimizer_including=cudnn\n[lib]\ncnmem=0.1\n[nvcc]\nfastmath=True\n[dnn]\nenabled=True\ninclude_path=/usr/local/cuda/include\nlibrary_path=/usr/local/cuda/lib64" > /root/.theanorc

# Download original code
RUN git clone https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling.git
RUN mv Generative-and-Discriminative-Voxel-Modeling vrnens
RUN git clone https://github.com/dimatura/voxnet.git

# Copy changed code
WORKDIR vrnens
COPY *.py Discriminative/
RUN mv Discriminative/layers.py /voxnet/voxnet
WORKDIR /vrnens/Discriminative
COPY *.ini ./
RUN mkdir logs
ENTRYPOINT /bin/bash 
