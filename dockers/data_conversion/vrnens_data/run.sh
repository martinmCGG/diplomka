##########################################################################################################
# Set required variables

name="vrnensa"
dataset_path="/local/krabec/ModelNet40A/ModelNet40A"
output_dir="/local/krabec/ModelNet40A"
docker_hidden=t

##########################################################################################################

image_name="openvdb"

output_dir="$output_dir/$name"
mkdir -m 777 $output_dir
docker build -t "$image_name" .
docker kill "$image_name"
docker rm "$image_name"

docker run --rm -id --name "$image_name" -v "$dataset_path":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "python vrnens_data.py"
#docker exec -it "$image_name" sh -c "python Modelnet.py"
##########################################################################################################
