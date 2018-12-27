docker build -t pnet .

dataset="/home/krabec/data/ModelNet40_pnet"
out="/home/krabec/dockers/pointnet/out"
docker run --runtime=nvidia --rm -id -v "$out":/pointnet/logs -v "$dataset":/data pnet