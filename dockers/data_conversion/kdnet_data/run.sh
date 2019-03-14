##########################################################################################################
# Set required variables

name="ShapeNet_kdnet"
#name="Small_converted"
#dataset_path="/local/krabec/ModelNet40A"
dataset_path="/home/krabec/data/ShapeNet"
output_dir="/local/krabec"
docker_hidden=d

##########################################################################################################

image_name="kdnet_data"

output_dir="$output_dir/$name"
mkdir -m 777 $output_dir
docker build -t "$image_name" .
docker kill "$image_name"
docker rm "$image_name"

docker run --rm -id --name "$image_name" -v "$dataset_path":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "python kdnet_data.py"

##########################################################################################################
