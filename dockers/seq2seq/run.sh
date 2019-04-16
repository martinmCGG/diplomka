##########################################################################################################
# Set required variables

name='seq2seq'
dataset_path="/local/krabec/ModelNet40A/shaded"
out_path="/media/user/TMP/repo/MK-diplomka-test/out/seq2seq"
GPU=0
docker_hidden=t

##########################################################################################################

mkdir  "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/seq2seq/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################
