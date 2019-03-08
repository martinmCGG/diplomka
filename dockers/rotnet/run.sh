##########################################################################################################
# Set required variables
name='rotnet'
dataset_path="/local/krabec/ModelNet40A_mvcnn_pbrt"
out_path="/home/krabec/dockers/rotnet/out/"
GPU=0
docker_hidden=t

##########################################################################################################

mkdir -r "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/rotationnet/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################

#docker exec -it cafferotnet bash -c "python prepare_data.py /data/train.txt" 
#docker exec -it cafferotnet bash -c "python prepare_data.py /data/test.txt --test" 
