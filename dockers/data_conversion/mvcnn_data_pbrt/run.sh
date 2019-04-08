##########################################################################################################
# Set required variables

name="pbrt"
dataset_path="/local/krabec/ModelNet40A"
output_dir="/local/krabec/"
docker_hidden=t

##########################################################################################################

image_name="pbrt"

output_dir="$output_dir/$name"
mkdir -m 777 $output_dir
docker build -t "$image_name" .
docker kill "$image_name" 2>/dev/null
docker rm "$image_name" 2>/dev/null

docker run --rm -id --name "$image_name" -v "$dataset_path":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "python3 mvcnn_data.py"

##########################################################################################################

if [ "$docker_hidden" == d ]; then echo Container running in detached mode. Check the log file for the information; fi
