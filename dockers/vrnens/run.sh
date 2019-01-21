docker build -t vrnens .
docker kill vrnens
docker rm vrnens

dataset="/home/krabec/data/ModelNet40_vrnens"
out="/home/krabec/dockers/vrnens/out"

docker run --runtime=nvidia --rm -id --name vrnens -v "$out":/vrnens/Discriminative/logs -v "$dataset":/data vrnens
docker exec -id vrnens sh -c "python train.py VRN.py /data/converted"
#docker exec -it vrnens sh -c "python test_ensemble.py VRN.py /data/converted --weights 2"