docker build -t rotnet_torch .
docker kill rotnet_torch
docker rm rotnet_torch

dataset="/home/krabec/data/ModelNet40A_mvcnn_12"
out="/home/krabec/dockers/rotnet_torch/out"

docker run --runtime=nvidia --rm  --shm-size 8G -id --name rotnet_torch -v "$out":/rotationnet/logs -v "$dataset":/data rotnet_torch

#docker exec -it rotnet_torch sh -c "python link_images.py /data/converted/train.txt /data/converted/test.txt --out /data/rotnet" 
docker exec -it rotnet_torch sh -c "export CUDA_VISIBLE_DEVICES=0,1,2 && python3 train_rotationnet.py --case 1 --pretrained -a alexnet -b 240 --lr 0.01 --epochs 1500 /data/rotnet| tee logs/log.txt"
#docker exec -it rotnet_torch sh -c "python test.py --weights 90"

