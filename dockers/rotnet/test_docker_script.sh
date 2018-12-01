#!/bin/bash
docker kill cafferotnet
docker rm cafferotnet
docker run --runtime=nvidia -id --name cafferotnet -v /home/krabec/models:/home/krabec/models caffe/rotnet:latest sh

#docker exec -it cafferotnet bash -c "cd /home/krabec/models/ROTNET && sh test_modelnet40.sh"
docker exec -it cafferotnet bash -c "cd /home/krabec/models/ROTNET && sh my_test_modelnet.sh"
#docker exec -it cafferotnet bash -c "cd /home/krabec/models/ROTNET && python save_scores.py --center_only --gpu --model_def deploy_modelnet40_case2.prototxt --pretrained_model rotationnet_modelnet40_case2.caffemodel --input_file ModelNet40v2/test_airplane.txt --output_file ModelNet40v2/test_airplane.npy"
#docker exec -it cafferotnet bash -c "cd /home/krabec/models/ROTNET && python my_classify_modelnet.py --center_only --model_def deploy_modelnet40_case2.prototxt --pretrained_model rotationnet_modelnet40_case2.caffemodel"
