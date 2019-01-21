docker build -t pnet2 .

docker kill pnet2
dataset="/home/krabec/data/ModelNet40_pnet"
out="/home/krabec/dockers/pointnet2/out"

docker run --runtime=nvidia --name pnet2 --rm -id -v "$out":/pointnet2/logs -v "$dataset":/data pnet2
#docker exec -it pnet2 sh -c "rm -rf logs/*"

docker exec -id pnet2 sh -c "python train.py --data /data/converted "
# docker exec -it pnet2 sh -c "python evaluate.py --data /data/converted --weights 10"