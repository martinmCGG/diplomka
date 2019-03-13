##########################################################################################################
# Set required variables

#name="ModelNet40A_octree"
name="Small_converted"
#dataset="/home/krabec/data/ShapeNet"
dataset="/local/krabec/Ssmall"
output_dir="/local/krabec"
docker_hidden=t

##########################################################################################################

image_name="octree"

output_dir="$output_dir/$name"
mkdir $output_dir
docker build -t "$image_name" .
docker kill "$image_name"
docker rm "$image_name"

docker run --rm -id --name "$image_name" -v "$dataset":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "python octree_data.py"

##########################################################################################################
