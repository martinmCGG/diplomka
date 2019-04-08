##########################################################################################################
# Set required variables

name="sonet5000"
dataset_path="/media/user/TMP/ModelNet40"
output_dir="/media/user/TMP/ModelNet40/converted_sonet"
GPU=0
docker_hidden=i

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
