docker build -t pbrt_w -f Dockerfile /home/krabec/dockers/pbrt
docker run -id --mount type=bind,source="/home/krabec/Data/ModelNet40Small",target=/data  --name pbrt pbrt_w 
docker exec -it pbrt bash -c "python3 mvcnn_data.py -o /data/images -d /data"
docker kill pbrt
docker rm pbrt
