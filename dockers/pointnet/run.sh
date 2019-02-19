docker build -t pnet .

docker kill pnet
#dataset="/home/krabec/data/ModelNet40A_pnet2"
dataset="/home/krabec/models/PNET2/data"

out="/home/krabec/dockers/pointnet/out"
docker run --runtime=nvidia --name pnet --rm -id -v "$out":/pointnet/logs -v "$dataset":/data pnet

#docker exec -it pnet sh -c "rm -rf logs/*"

#docker exec -it pnet sh -c "export CUDA_VISIBLE_DEVICES=1 && python train.py --data /data/converted"
docker exec -it pnet sh -c "export CUDA_VISIBLE_DEVICES=1 && python evaluate.py --data /data/converted --weights 80"

