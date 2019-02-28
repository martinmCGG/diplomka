docker build -t seq2seq .

docker kill seq2seq
dataset="/local/krabec/ModelNet40A_mvcnn_depth"
out="/home/krabec/dockers/seq2seq/out/depth"

docker run --runtime=nvidia --name seq2seq --rm -id -v "$out":/seq2seq/logs -v "$dataset":/data seq2seq
#docker exec -it seq2seq sh -c "rm -rf logs/*"

docker exec -id seq2seq sh -c "export CUDA_VISIBLE_DEVICES=0 && python run.py --data /data"
#docker exec -it seq2seq sh -c "export CUDA_VISIBLE_DEVICES=0 && python run.py --test --data /data --weights 200"

