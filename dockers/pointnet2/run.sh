docker build -t pnet2 .

docker kill pnet2
#dataset="/home/krabec/data/ModelNet40A_pnet2"
dataset="/home/krabec/models/PNET2/data"
out="/home/krabec/dockers/pointnet2/out"

docker run --runtime=nvidia --name pnet2 --rm -id -v "$out":/pointnet2/logs -v "$dataset":/data pnet2
#docker exec -it pnet2 sh -c "rm -rf logs/*"

#docker exec -it pnet2 sh -c "export CUDA_VISIBLE_DEVICES=3 && python train.py --data /data/converted "
docker exec -it pnet2 sh -c "export CUDA_VISIBLE_DEVICES=3 && python evaluate.py --data /data/converted --weights 140"
