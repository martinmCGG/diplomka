##########################################################################################################
# Set required variables
name='rotnet'
dataset_path="/local/krabec/ModelNet40A/phong"
out_path="/home/krabec/dockers/rotnet/out/phong"
GPU=2
docker_hidden=d

##########################################################################################################

mkdir  "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/rotationnet/logs -v "$dataset_path":/data "$name"
docker exec -it "$name" bash -c "python prepare_data.py /data/train.txt" 
docker exec -it "$name" bash -c "python prepare_data.py /data/test.txt --test" 
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################


