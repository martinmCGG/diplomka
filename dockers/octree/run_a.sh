docker build -t octree_a .

docker kill octree_a
docker rm octree_a
dataset="/local/krabec/ModelNet40A_octree_a"
out="/home/krabec/dockers/octree/out_a/"

docker run --runtime=nvidia --name octree_a --rm -id -v "$out":/workspace/logs -v "$dataset":/data octree_a

docker exec -it octree_a sh -c "export CUDA_VISIBLE_DEVICES=0 && python train.py --solver ao-cnn/cls_5.solver.prototxt --data /data"
#docker exec -it octree_a sh -c "export CUDA_VISIBLE_DEVICES=0 && python train.py --test --solver ao-cnn/cls_5.solver.prototxt --data /data --weights 4000"

