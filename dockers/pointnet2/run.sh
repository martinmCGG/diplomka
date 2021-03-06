##########################################################################################################
# Set required variables

name='pnet2'
dataset_path="/local/krabec/ShapeNet/pnet"
out_path="/home/krabec/dockers/pointnet2/shapenet/"
GPU=3
docker_hidden=d

##########################################################################################################

mkdir "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/pointnet2/logs -v "$dataset_path":/data "$name"

docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################
