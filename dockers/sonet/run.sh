##########################################################################################################
# Set required variables

name='sonet'
dataset_path="/local/krabec/ModelNet40A/sonet5000"
out_path="/media/user/TMP/repo/MK-diplomka-test/out/sonet"
GPU=0
docker_hidden=t

##########################################################################################################

mkdir "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/sonet/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################

#docker exec -it "$name" sh -c "rm -rf /sonet/logs*"