#!/bin/bash

docker kill cafferotnet
docker rm cafferotnet
docker build -t cafferotnet .

dataset="/home/krabec/data/ModelNet40A_mvcnn_12"
out="/home/krabec/dockers/rotnet/out"

docker run --runtime=nvidia --rm -id --name cafferotnet -v "$out":/rotationnet/logs -v "$dataset":/data cafferotnet:latest sh
docker exec -it cafferotnet bash -c "python prepare_data.py /data/converted/train.txt" 
docker exec -it cafferotnet bash -c "python prepare_data.py /data/converted/test.txt" 
#docker exec -it cafferotnet bash -c "export CUDA_VISIBLE_DEVICES=0 && python train.py --solver Training/rotationnet_modelnet40_case1_solver.prototxt 2>&1 | tee logs/log.txt"
docker exec -it cafferotnet bash -c "/opt/caffe/caffe-rotationnet2/build/tools/caffe train -solver Training/rotationnet_modelnet40_case1_solver.prototxt -weights caffe_nets/ilsvrc13 -gpu 1 2>&1 | tee logs/log.txt" 