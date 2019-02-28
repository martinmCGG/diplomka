docker build -t octree .

docker kill octree
dataset="/local/krabec/ModelNet40A_octree"
out="/home/krabec/dockers/octree/out/"

docker run --runtime=nvidia --name octree --rm -id -v "$out":/workspace/logs -v "$dataset":/data octree

docker exec -id octree bash
#docker exec -id octree sh -c "export CUDA_VISIBLE_DEVICES=0 && python run.py --data /data"
#docker exec -it octree sh -c "export CUDA_VISIBLE_DEVICES=0 && python run.py --test --data /data --weights 280"

