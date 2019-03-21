##########################################################################################################
# Set required variables

name="depth"
dataset_path="/local/krabec/ShapeNet/ShapeNet"
output_dir="/local/krabec/ShapeNet"
docker_hidden=t

#This must be one of shaded or depth
render=shaded

##########################################################################################################

image_name="blender"

output_dir="$output_dir/$name"
mkdir -m 777 $output_dir
docker build -t "$image_name" .
docker kill "$image_name"
docker rm "$image_name"

docker run --rm -id --name "$image_name" -v "$dataset_path":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "/usr/local/blender/blender render_$render.blend -noaudio -b -P  blender_data.py" 

##########################################################################################################
