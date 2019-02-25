docker build -t pbrt_w .

#Path to the dataset
dataset="/home/krabec/data/ModelNet40A"
output_dir="/home/krabec/data/ModelNet40A_pbrt"
#Type of the dataset. Currently must be one of (modelnet, shapenet)
dataset_type="modelnet"

#To see other options run python script with -h option and change docker run parameter to -d to -it.
docker run --rm -id -v "$dataset":/dataset -v "$output_dir":/data pbrt_w \
sh -c "python3 mvcnn_data.py /dataset /data -v 12  --dataset $dataset_type"

