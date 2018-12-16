docker build -t pointcloud .

#Path to the dataset
dataset="/home/krabec/Data/ModelNet40Small"
#Type of the dataset. Currently must be one of (modelnet, shapenet)
dataset_type="modelnet"

#To see other options run python script with -h option and change docker run parameter to -d to -it.
docker run --rm -d --mount type=bind,source="$dataset",target=/data pointcloud \
sh -c "python3 pnet_data.py /data data/converted --dataset $dataset_type"