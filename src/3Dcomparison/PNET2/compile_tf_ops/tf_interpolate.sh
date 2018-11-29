
TF_INC=$(/usr/local/lib/python2.7/dist-packages/tensorflow/include)
echo $TF_INC
TF_LIB=$(/usr/local/lib/python2.7/dist-packages/tensorflow)
echo $TF_LIB
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
