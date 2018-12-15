docker build -t pbrt_w .

docker run --rm -it --mount type=bind,source="/home/krabec/Data/ShapeNetSmall",target=/data pbrt_w \
sh -c "python3 mvcnn_data.py /data /data/images2 --dodecahedron --dataset shapenet"
