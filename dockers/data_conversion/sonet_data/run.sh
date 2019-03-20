##########################################################################################################
# Set required variables

name="sonet"
dataset_path="/local/krabec/ShapeNet"
output_dir="/home/krabec"
GPU=2
docker_hidden=t

##########################################################################################################

image_name="sonet"

output_dir="$output_dir/$name"
mkdir $output_dir
docker build -t "$image_name" .
docker kill "$image_name"
docker rm "$image_name"

docker run --runtime=nvidia --rm -id --name "$image_name" -v "$dataset_path":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python3 sonet_data.py"

##########################################################################################################
