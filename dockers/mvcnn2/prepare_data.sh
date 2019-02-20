docker build -t mvcnn2 .
docker kill mvcnn2

dataset="/local/krabec/Modelnet40A_mvcnn12_shaded"
out="/local/krabec"
docker run --runtime=nvidia --rm -id --name mvcnn2 -v "$out":/mvcnn2/logs -v "$dataset":/out mvcnn2
docker exec -it mvcnn2 sh -c "python prepare_data.py"

