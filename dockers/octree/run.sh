docker build -t octree .

docker kill octree
docker rm octree
dataset="/local/krabec/ModelNet40A_octree"
out="/home/krabec/dockers/octree/out/"

docker run --runtime=nvidia --name octree --rm -id -v "$out":/workspace/logs -v "$dataset":/data octree

docker exec -id octree sh -c "export CUDA_VISIBLE_DEVICES=1 && python train.py --solver o-cnn/solver_M40_5.prototxt --data /data"
#docker exec -it octree sh -c "export CUDA_VISIBLE_DEVICES=1 && python train.py --test --solver o-cnn/solver_M40_5.prototxt --data /data --weights 4000"

