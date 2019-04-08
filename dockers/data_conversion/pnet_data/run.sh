##########################################################################################################
# Set required variables

name="pnet"
dataset_path="/media/user/TMP/ModelNet40"
output_dir="/media/user/TMP/ModelNet40/converted_pnet"
docker_hidden=t

##########################################################################################################

image_name="pointcloud"

output_dir="$output_dir/$name"
mkdir -m 777 $output_dir
docker build -t "$image_name" .
docker kill "$image_name"
docker rm "$image_name"

docker run --rm -id --name "$image_name" -v "$dataset_path":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "python3 pnet_data.py"

##########################################################################################################
