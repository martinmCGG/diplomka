#!/bin/bash
docker kill cafferotnet
docker rm cafferotnet
docker run -id --name cafferotnet -v /home/krabec/models:/models caffe/rotnet:latest sh
docker exec -it cafferotnet bash -c "ls"
docker exec -it cafferotnet bash -c "cd models/ROTNET && sh test_full_modelnet40.sh"
