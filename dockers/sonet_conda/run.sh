docker build -t sonet_conda .
dataset="/home/krabec/data/ModelNet40A_pnet"
out="/home/krabec/dockers/sonet_conda/Out"
docker run --runtime=nvidia -it -v "$out":/sonet/logs -v "$dataset":/data sonet_conda
