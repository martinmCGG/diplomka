#!/bin/bash
docker kill theano1
docker rm theano1
nvidia-docker run -id --name theano1 -v /home/krabec/models:/models theanocuda:latest sh
nvidia-docker exec -it theano1 bash -c "ls"
nvidia-docker exec -it theano1 bash -c "cd ./models/VRNENS/Discriminative && python test_ensemble.py ensemble_model1.py"
