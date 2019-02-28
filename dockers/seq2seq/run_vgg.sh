docker build -t vgg .
docker kill vgg
dataset="/local/krabec/ModelNet40A_mvcnn_pbrt"
out="/home/krabec/dockers/seq2seq/out/pbrt"

docker run --runtime=nvidia --name vgg --rm -id -v "$out":/seq2seq/logs -v "$dataset":/data vgg
docker exec -id vgg sh -c "export CUDA_VISIBLE_DEVICES=1 && python train_vgg.py --data /data --weights 20"
#docker exec -it vgg sh -c "export CUDA_VISIBLE_DEVICES=1 && python evaluate_vgg.py --data /data --weights 40"
#docker exec -it vgg sh -c "export CUDA_VISIBLE_DEVICES=1 && python train_vgg.py --data /data --extract --weights 40"
