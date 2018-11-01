#!/bin/bash
docker kill cafferotnet
docker rm cafferotnet
docker run --runtime=nvidia -id --name cafferotnet -v /home/krabec/models:/home/krabec/models caffe/rotnet:latest sh
docker exec -it cafferotnet bash -c "ls"
#docker exec -it cafferotnet bash -c "cd /home/krabec/models/ROTNET && sh test_modelnet40.sh"
#docker exec -it cafferotnet bash -c "cd /home/krabec/models/ROTNET && python save_scores.py --center_only --mean_file=VGG_mean.npy --gpu --model_def deploy_modelnet40_case1.prototxt --pretrained_model rotationnet_modelnet40_case1.caffemodel --input_file ModelNet40v1/test_airplane.txt --output_file ModelNet40v1/test_airplane.npy"
#docker exec -it cafferotnet bash -c "cd /home/krabec/models/ROTNET && python my_classify_modelnet.py --center_only --model_def deploy_modelnet40_case1.prototxt --pretrained_model rotationnet_modelnet40_case1.caffemodel"
docker exec -it cafferotnet bash -c "cd /home/krabec/models/ROTNET && /opt/caffe/caffe-rotationnet2/build/tools/caffe train -solver Training/rotationnet_modelnet40_case1_solver.prototxt -weights caffe_nets/ilsvrc13 2>&1 | tee log.txt" 