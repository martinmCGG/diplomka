#!/bin/bash
docker kill cafferotnet
docker rm cafferotnet
docker run --runtime=nvidia -id --name cafferotnet -v /home/krabec/models:/home/krabec/models caffe/rotnet:latest sh
docker exec -it cafferotnet bash -c "cd /home/krabec/models/ROTNET && /opt/caffe/caffe-rotationnet2/build/tools/caffe train -solver Training/rotationnet_modelnet40_case2_solver.prototxt -weights caffe_nets/ilsvrc13 2>&1 | tee log.txt" 