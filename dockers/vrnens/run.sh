##########################################################################################################
# Set required variables

name='vrnens2'
dataset_path="/local/krabec/ModelNet40A/vrnens"
out_path="/home/krabec/dockers/vrnens2/out/"
GPU=1
docker_hidden=t

##########################################################################################################

mkdir  "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/vrnens/Discriminative/logs -v "$dataset_path":/data "$name"

docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################

