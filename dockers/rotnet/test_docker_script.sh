#!/bin/bash

docker build -t cafferotnet .
docker kill cafferotnet
docker rm cafferotnet

dataset="/home/krabec/data/ModelNet40_mvcnn"
out="/home/krabec/dockers/rotnet/out"

docker run --runtime=nvidia --rm -id --name cafferotnet -v "$out":/rotationnet/logs -v "$dataset":/data cafferotnet:latest sh

docker exec -it cafferotnet bash -c "python my_classify_modelnet.py"

