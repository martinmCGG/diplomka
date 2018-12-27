docker build -t pnet2 .

dataset="/home/krabec/Data/ModelNet40Small_pnet_normals"

docker run --runtime=nvidia --rm -it --mount type=bind,source="$dataset",target=/data pnet2