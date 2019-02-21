docker build -t mvcnn2 .
docker kill mvcnn2

dataset="/local/krabec/Modelnet40A_mvcnn2"
out="/home/krabec/dockers/mvcnn2/out"

docker run --runtime=nvidia --rm -id --name mvcnn2 -v "$out":/mvcnn2/logs -v "$dataset":/out mvcnn2

docker exec -id mvcnn2 sh -c "export CUDA_VISIBLE_DEVICES=3 && python train_mvcnn.py -name mvcnn -num_models 1000 -weight_decay 0.001 -num_views 12 -cnn_name vgg11"
#docker exec -it mvcnn2 sh -c "export CUDA_VISIBLE_DEVICES=3 && python train_mvcnn.py -name mvcnn --test --weights 30 -num_views 12"