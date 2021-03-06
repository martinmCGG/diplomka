##########################################################################################################
# Set required variables

name='pnet'
dataset_path="/local/krabec/ModelNet40A/pnet"
out_path="/home/krabec/dockers/pointnet/out2/"
GPU=2
docker_hidden=t

##########################################################################################################

mkdir "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/pointnet/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################
