docker build -t octree .

docker kill octree
dataset="/local/krabec/ModelNet40A"
out="/local/krabec/ModelNet40A_octree"

docker run --runtime=nvidia --name octree --rm -id -v "$out":/data -v "$dataset":/dataset octree

docker exec -it octree sh -c "python octree_data.py /dataset /data"
