##########################################################################################################
# Set required variables

name='seq2seq'
dataset_path="/local/krabec/ModelNet40A_mvcnn_shaded"
out_path="/home/krabec/dockers/seq2seq/out/"
GPU=1
docker_hidden=t

##########################################################################################################

mkdir  "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/seq2seq/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################
