docker build -t mvcnn .
dataset="/home/krabec/data/ModelNet40_mvcnn"

docker run --runtime=nvidia -d --mount type=bind,source="$dataset",target=/data mvcnn