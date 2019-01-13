docker build -t pnet .

docker kill pnet
dataset="/home/krabec/data/ModelNet40_pnet"
out="/home/krabec/dockers/pointnet/out"
docker run --runtime=nvidia --name pnet --rm -id -v "$out":/pointnet/logs -v "$dataset":/data pnet
#docker exec -it pnet sh -c "rm -rf logs/*"

docker exec -id pnet sh -c "python train.py --data /data/converted"

#docker exec -it pnet sh -c "python evaluate.py --data /data/converted --weights 60"