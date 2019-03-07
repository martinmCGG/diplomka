##########################################################################################################
# Set required variables

name='octree_adaptive'
dataset_path="/local/krabec/ModelNet40A_octree_a"
out_path="/home/krabec/dockers/octree_adaptive/out/"
GPU=0
docker_hidden=t

##########################################################################################################

mkdir -r "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/workspace/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################
