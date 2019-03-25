##########################################################################################################
# Set required variables

name='mvcnn2'
dataset_path="/local/krabec/ModelNet40A/depth"
out_path="/home/krabec/dockers/mvcnn2/a/"
GPU=0
docker_hidden=t

##########################################################################################################

mkdir "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/mvcnn2/logs -v "$dataset_path":/data "$name"

docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################
