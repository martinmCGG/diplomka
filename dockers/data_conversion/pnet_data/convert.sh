docker build -t pointcloud .

#Path to the dataset
dataset="/home/krabec/data/ModelNet40A"
output_dir="/home/krabec/data/ModelNet40A_pnet"
#Type of the dataset. Currently must be one of (modelnet, shapenet)
dataset_type="modelnet"


#To see other options run python script with -h option and change docker run parameter to -d to -it.
docker run --rm -it -v "$dataset":/dataset -v "$output_dir":/data pointcloud \
sh -c "python3 pnet_data.py /dataset /data --dataset $dataset_type"