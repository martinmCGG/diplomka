docker build -t kdnet_theano .
docker kill kdnet_theano
docker rm kdnet_theano

dataset="/home/krabec/data/ModelNet40A_kdnet"
out="/home/krabec/dockers/kdnet_theano/out"

docker run --runtime=nvidia --rm -id --name kdnet_theano -v "$out":/kdnets/logs -v "$dataset":/data kdnet_theano
#docker exec -it kdnet_theano sh -c "python prepare_data.py"
#docker exec -it kdnet_theano sh -c "python train.py --data /data --weights 95 "
docker exec -it kdnet_theano sh -c "python train.py --data /data --test --weights 140 "
