docker build -t seq2seq .

docker kill seq2seq
dataset="/home/krabec/data/ModelNet40A_mvcnn12"
out="/home/krabec/dockers/seq2seq/out"

docker run --runtime=nvidia --name seq2seq --rm -id -v "$out":/seq2seq/logs -v "$dataset":/data seq2seq
#docker exec -it seq2seq sh -c "rm -rf logs/*"

docker exec -it seq2seq sh -c "python run.py --train --data /data/converted"
#docker exec -it seq2seq sh -c "python run.py --data /data/converted --weights 50"
