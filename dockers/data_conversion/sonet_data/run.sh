##########################################################################################################
# Set required variables

name="sonet5000"
dataset_path="/local/krabec/ModelNet40A/ModelNet40A"
output_dir="/local/krabec/ModelNet40A/"
GPU=2
docker_hidden=d

##########################################################################################################

image_name="sonet"

output_dir="$output_dir/$name"
mkdir -p "$output_dir"
docker build -t "$image_name" .
docker kill "$image_name" 2>/dev/null
docker rm "$image_name" 2>/dev/null
    
docker run --runtime=nvidia --rm -id --name "$image_name" -v "$dataset_path":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python3 sonet_data.py"

##########################################################################################################

if [ "$docker_hidden" == d ]; then echo Container running in detached mode. Check the log file for the information; fi
