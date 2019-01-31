docker build -t kdnet .
docker kill kdnet

dataset="/home/krabec/data/ModelNet40A_pnet2"
out="/home/krabec/dockers/kdnet/out"

docker run --runtime=nvidia --name kdnet --rm -id -v "$out":/kdnet/logs -v "$dataset":/data kdnet

docker exec -it kdnet sh -c "export CUDA_VISIBLE_DEVICES=0 && python train_batch.py --data /data/converted --weights 0" 
#docker exec -it kdnet sh -c "export CUDA_VISIBLE_DEVICES=0 && python train_batch.py --test --data /data/converted --weights 0"
