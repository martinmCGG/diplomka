##########################################################################################################
# Set required variables

name='octree'
dataset_path="/media/user/TMP/ModelNet40/converted_octree/octree/"
out_path="/media/user/TMP/repo/MK-diplomka-test/out/octree"
GPU=0
docker_hidden=t

##########################################################################################################

mkdir "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/workspace/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################