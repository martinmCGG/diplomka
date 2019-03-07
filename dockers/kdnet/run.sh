##########################################################################################################
# Set required variables

name='kdnet'
dataset_path="/local/krabec/ModelNet40A_kdnet"
out_path="/home/krabec/dockers/kdnet/out/"
GPU=2
docker_hidden=d

##########################################################################################################

mkdir -r "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/kdnets/logs -v "$dataset_path":/data "$name"

#docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################


docker exec -i -"$docker_hidden" "$name" sh -c "rm -rf /kdnets/logs/*"

