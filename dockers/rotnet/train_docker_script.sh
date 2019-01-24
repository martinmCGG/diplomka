#!/bin/bash

docker build -t cafferotnet .
docker kill cafferotnet
docker rm cafferotnet

dataset="/home/krabec/data/ModelNet40_mvcnn12"
out="/home/krabec/dockers/rotnet/out"

docker run --runtime=nvidia --rm -id --name cafferotnet -v "$out":/rotationnet/logs -v "$dataset":/data cafferotnet:latest sh
#docker exec -it cafferotnet bash
docker exec -id cafferotnet bash -c "/opt/caffe/caffe-rotationnet2/build/tools/caffe train -solver Training/rotationnet_modelnet40_case2_solver.prototxt -weights caffe_nets/ilsvrc13 -gpu 1 2>&1 | tee logs/log.txt" 