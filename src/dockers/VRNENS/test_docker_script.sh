#!/bin/bash
docker kill theano1
docker rm theano1
nvidia-docker run --runtime=nvidia -id --name theano1 -v /home/krabec/models:/home/krabec/models theanocuda:latest sh
#nvidia-docker exec -it theano1 bash -c "cd /home/krabec/models/voxnet && pip install --editable ."
nvidia-docker exec -it theano1 bash -c "cd /home/krabec/models/VRNENS/Discriminative && python test_ensemble.py ensemble_models"
