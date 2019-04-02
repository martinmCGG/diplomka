	##########################################################################################################
# Set required variables
name='mvcnn'
dataset_path="/local/krabec/ShapeNet/shaded"
out_path="/home/krabec/dockers/mvcnn/shapenet"
GPU=1
docker_hidden=d
##########################################################################################################
mkdir "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"
docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/mvcnn/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"
##########################################################################################################
