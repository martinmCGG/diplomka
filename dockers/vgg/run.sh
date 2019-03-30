##########################################################################################################
# Set required variables

name='vgg'
dataset_path="/local/krabec/ModelNet40A/shaded"
out_path="/home/krabec/dockers/vgg/out/"
GPU=2
docker_hidden=d

##########################################################################################################

mkdir -r "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/vgg/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################
