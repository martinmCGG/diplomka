docker build -t openvdb .

docker run --rm -it --mount type=bind,source="/home/krabec/Data/ShapeNetSmall",target=/data openvdb \
sh -c "python vrnens_data.py /data data/voxels --dataset shapenet"
