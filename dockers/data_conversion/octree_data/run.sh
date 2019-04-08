##########################################################################################################
# Set required variables

name="octree_adaptive"
dataset_path="/local/krabec/ShapeNet/"
output_dir="/home/krabec/"
docker_hidden=d

##########################################################################################################

image_name="octree"

output_dir="$output_dir/$name"
mkdir -p "$output_dir"
docker build -t "$image_name" .
docker kill "$image_name" 2>/dev/null
docker rm "$image_name" 2>/dev/null

docker run --rm -id --name "$image_name" -v "$dataset_path":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "python octree_data.py > /dataset/log.txt"

##########################################################################################################

if [ "$docker_hidden" == d ]; then echo Container running in detached mode. Check the log file for the information; fi
