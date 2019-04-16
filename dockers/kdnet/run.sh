##########################################################################################################
# Set required variables

name='kdnet'
dataset_path="/media/user/TMP/ModelNet40/converted_kdnet/kdnetsmall/"
out_path="/media/user/TMP/repo/MK-diplomka-test/out/kdnet"
GPU=0
docker_hidden=t

##########################################################################################################

mkdir "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/kdnets/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" nvidia-smi
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################


#docker exec -i -"$docker_hidden" "$name" sh -c "rm -rf /kdnets/logs/*"

