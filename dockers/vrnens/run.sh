docker build -t vrnens .
dataset="/home/krabec/Data/ModelNet40Small_vrnens"

docker run --runtime=nvidia --rm -it --mount type=bind,source="$dataset",target=/data vrnens