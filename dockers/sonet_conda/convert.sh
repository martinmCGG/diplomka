docker build -t sonet_conda .

#Path to the dataset
dataset="/home/krabec/Data/ModelNet1A"
#Type of the dataset. Currently must be one of (modelnet, shapenet)
dataset_type="modelnet"

#To see other options run python script with -h option and change docker run parameter to -d to -it.
docker run --rm -it --mount type=bind,source="$dataset",target=/data sonet_conda \
sh -c "python sonet_data/sonet_data.py /data /data/converted --dataset $dataset_type"