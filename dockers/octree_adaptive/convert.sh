docker build -t octree_a .

docker kill octree_a
docker rm octree_a
dataset="/local/krabec/ModelNet40A"
out="/local/krabec/ModelNet40A_octree_a"

docker run --runtime=nvidia --name octree_a --rm -id -v "$out":/data -v "$dataset":/dataset octree_a

docker exec -id octree_a sh -c "python octree_data.py /dataset /data --adaptive"
