##########################################################################################################
# Set required variables

name="vrnens"
dataset_path="/local/krabec/ShapeNet"
output_dir="/home/krabec"
docker_hidden=d

##########################################################################################################

image_name="openvdb"

output_dir="$output_dir/$name"
mkdir -m 777 $output_dir
docker build -t "$image_name" .
docker kill "$image_name"
docker rm "$image_name"

docker run --rm -id --name "$image_name" -v "$dataset_path":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "python vrnens_data.py"

##########################################################################################################
