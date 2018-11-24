FROM pythoncore

ENV CLONE_TAG=master

RUN wget http://zlib.net/zlib-1.2.11.tar.gz
RUN tar zxf zlib-1.2.11.tar.gz
RUN cd zlib-1.2.11 && ./configure && make -j 32 && make install

RUN wget https://github.com/01org/tbb/releases/download/2018_U6/tbb2018_20180822oss_lin.tgz
RUN dtrx tbb2018_20180822oss_lin.tgz

RUN wget https://github.com/openexr/openexr/releases/download/v2.3.0/ilmbase-2.3.0.tar.gz
RUN tar zxf ilmbase-2.3.0.tar.gz
RUN cd ilmbase-2.3.0 && autoconf && ./configure --prefix="/usr/include/openexr" && make -j 32 && make install

RUN wget https://github.com/openexr/openexr/releases/download/v2.3.0/openexr-2.3.0.tar.gz
RUN tar zxf openexr-2.3.0.tar.gz
RUN cd openexr-2.3.0 && autoconf && ./configure --prefix="/usr/include/openexr" --with-ilmbase-prefix="/usr/include/openexr"  && make -j 32 && make install

#RUN git clone -b ${CLONE_TAG} --recursive https://github.com/log4cplus/log4cplus.git
#RUN cd log4cplus && ./configure && make && make dist && make install

#RUN git clone https://github.com/doxygen/doxygen.git
#RUN cd doxygen && cmake -G "Unix Makefiles" . && make -j 32 . && make install .

RUN wget https://github.com/jemalloc/jemalloc/releases/download/4.0.3/jemalloc-4.0.3.tar.bz2
RUN dtrx jemalloc-4.0.3.tar.bz2 && mv jemalloc-4.0.3 jemalloc
RUN cd jemalloc && autoconf && ./configure && make -j 32 && make install

RUN wget http://dev-www.libreoffice.org/src/cppunit-1.14.0.tar.gz
RUN dtrx cppunit-1.14.0.tar.gz && mv cppunit-1.14.0 cppunit
RUN cd cppunit && ./configure && make -j 32 && make install

RUN wget https://github.com/glfw/glfw/releases/download/3.2.1/glfw-3.2.1.zip
RUN dtrx glfw-3.2.1.zip
RUN cd glfw-3.2.1 && cmake -DBUILD_SHARED_LIBS=ON . && make -f Makefile -j 32 && make -f Makefile install

RUN wget https://github.com/AcademySoftwareFoundation/openvdb/archive/v5.2.0.zip
RUN echo "H" | dtrx v5.2.0.zip 
RUN mv openvdb-5.2.0 openvdb

#RUN git clone https://github.com/Blosc/c-blosc.git
#RUN cd c-blosc && cmake .
#RUN cd /usr/include/c-blosc && cmake --build . -- -j 32 && cmake --build . --target install

RUN git clone https://github.com/Alobal123/diplomka.git
RUN cd diplomka/dockers/OPENVDB && cp Makefile ../../openvdb/openvdb/Makefile

RUN ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py27.a /usr/lib/x86_64-linux-gnu/libboost_python2.7.a
RUN ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py27.so /usr/lib/x86_64-linux-gnu/libboost_python2.7.so

RUN cd openvdb/openvdb && make -j 32 . && make install -j 32 .

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH":/usr/local/openvdb/lib/:/usr/include/ilmbase-2.3.0/Half/.libs:/usr/local/lib/"
ENV PYTHONPATH=$PYTHONPATH:/usr/local/openvdb/python/lib/python2.7/
ENTRYPOINT /bin/bash 