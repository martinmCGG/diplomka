##########################################################################################################
# Set required variables

name='seq2seq'
dataset_path="/local/krabec/ModelNet40A/shaded"
out_path="/home/krabec/dockers/seq2seq/out/shaded2"
GPU=0
docker_hidden=d

##########################################################################################################

mkdir  "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/seq2seq/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################
