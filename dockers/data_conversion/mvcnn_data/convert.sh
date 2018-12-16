docker build -t pbrt_w .

#Path to the dataset
dataset="/home/krabec/Data/ModelNet40Small"
#Type of the dataset. Currently must be one of (modelnet, shapenet)
dataset_type="modelnet"

#To see other options run python script with -h option and change docker run parameter to -d to -it.
docker run --rm -d --mount type=bind,source="$dataset",target=/data pbrt_w \
sh -c "python3 mvcnn_data.py /data /data/converted --dodecahedron --dataset $dataset_type"
