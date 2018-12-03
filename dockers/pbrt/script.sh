docker build -t pbrt_w -f Dockerfile /home/krabec/dockers/pbrt
docker run -id -v /home/krabec/Data/ShapeNetSmall:/data/ --name pbrt pbrt_w
docker exec -it pbrt bash -c "python3 mesh_to_images.py"
docker kill pbrt
docker rm pbrt
