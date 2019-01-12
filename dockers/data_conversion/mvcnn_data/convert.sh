docker build -t pbrt_w .

#Path to the dataset
dataset="/home/krabec/data/ModelNet40"

#Type of the dataset. Currently must be one of (modelnet, shapenet)
dataset_type="modelnet"

#To see other options run python script with -h option and change docker run parameter to -d to -it.
docker run --rm -it -v "$dataset":/data pbrt_w \
sh -c "python3 mvcnn_data.py /data /data/converted -v 12 --dataset $dataset_type"

