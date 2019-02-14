docker build -t seq2seq .

docker kill seq2seq
dataset="/home/krabec/data/ModelNet40A_mvcnn_12"
#dataset="/home/krabec/Data/ModelNet1A"
out="/home/krabec/dockers/seq2seq/out"

docker run --runtime=nvidia --name seq2seq --rm -id -v "$out":/seq2seq/logs -v "$dataset":/data seq2seq
#docker exec -it seq2seq sh -c "rm -rf logs/*"

#docker exec -id seq2seq sh -c "export CUDA_VISIBLE_DEVICES=3 && python train_vgg.py --data /data/converted"
docker exec -id seq2seq sh -c "export CUDA_VISIBLE_DEVICES=3 && python evaluate_vgg.py --data /data/converted --weights 2000"
#docker exec -id seq2seq sh -c "export CUDA_VISIBLE_DEVICES=3 && python train_vgg.py --data /data/converted --extract --weights 2000"
#docker exec -id seq2seq sh -c "export CUDA_VISIBLE_DEVICES=3 && python run.py --data /data/converted"
#docker exec -id seq2seq sh -c "export CUDA_VISIBLE_DEVICES=3 && python run.py --test --data /data/converted"
