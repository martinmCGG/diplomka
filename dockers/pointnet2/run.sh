docker build -t pnet2 .

dataset="/home/krabec/data/ModelNet40_pnet"
out="/home/krabec/dockers/pointnet2/out"

docker run --runtime=nvidia --rm -id -v "$out":/pointnet2/logs -v "$dataset":/data pnet2