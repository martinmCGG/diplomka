docker build -t sonet_conda .

#dataset="/home/krabec/data/ModelNet40A_pnet2"
dataset="/home/krabec/data/ModelNet40A_sonet"
out="/home/krabec/dockers/sonet_conda/Out"

docker kill sonet_conda
docker rm sonet_conda

docker run --runtime=nvidia --rm -id --name sonet_conda -v "$out":/sonet/logs -v "$dataset":/data sonet_conda
docker exec -it sonet_conda sh -c "python train.py"

