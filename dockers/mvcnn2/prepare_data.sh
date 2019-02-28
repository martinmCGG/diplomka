docker build -t mvcnn2 .
docker kill mvcnn2

dataset="/local/krabec/Modelnet40A_mvcnn_depth"
out="/local/krabec/Modelnet40A_mvcnn2_depth"
docker run --runtime=nvidia --rm -id --name mvcnn2 -v "$out":/out -v "$dataset":/data mvcnn2
docker exec -it mvcnn2 sh -c "python prepare_data.py"
