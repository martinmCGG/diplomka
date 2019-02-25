docker build -t blender .
docker kill blender

#Path to the dataset
dataset="/home/krabec/data/ModelNet40A"
output_dir="/local/krabec/Modelnet40A_mvcnn_depth"

#render="render_shaded.blend"
render="render_depth.blend"

#Type of the dataset. Currently must be one of (modelnet, shapenet)
dataset_type="modelnet"

#To see other options run python script with -h option and change docker run parameter to -d to -it.

docker run --rm -id --name blender -v "$dataset":/dataset -v "$output_dir":/data blender

docker exec -it blender sh -c "/usr/local/blender/blender $render -noaudio -b -P  blender_data.py -- -d /dataset -o /data -v 12 --dataset $dataset_type" 

docker kill blender