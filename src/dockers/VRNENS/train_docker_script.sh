#!/bin/bash
#docker kill theano1
#docker rm theano1
#nvidia-docker run --runtime=nvidia -id --name theano1 -v /home/krabec/models://home/krabec/models theanocuda:latest sh
nvidia-docker exec -it theano1 bash -c "cd /home/krabec/models/VRNENS && python Discriminative/train.py Discriminative/ensemble_model1.py datasets/modelnet40_rot_train.npz"
