docker build -t mvcnn .
docker kill mvcnn

dataset="/home/krabec/data/ModelNet40A_mvcnn_12"
out="/home/krabec/dockers/mvcnn/out"

docker run --runtime=nvidia --rm -id --name mvcnn -v "$out":/mvcnn/logs -v "$dataset":/data mvcnn
#docker exec -it mvcnn sh -c "python train.py"
docker exec -it mvcnn sh -c "python test.py --weights 90"