docker build -t sonet_conda .

#Path to the dataset
#dataset="/home/krabec/Data/ModelNet1A"
#dataset="/home/krabec/data/ModelNet40A"
dataset="/home/krabec/data/ModelNet40A_sonet"
#Type of the dataset. Currently must be one of (modelnet, shapenet)
dataset_type="modelnet"

docker kill sonet_conda
docker rm sonet_conda

#To see other options run python script with -h option and change docker run parameter to -d to -it.

docker run --runtime=nvidia --rm -id --name sonet_conda --mount type=bind,source="$dataset",target=/data sonet_conda

docker exec -it sonet_conda sh -c "python sonet_data/sonet_data.py /data /data/converted --dataset $dataset_type"
