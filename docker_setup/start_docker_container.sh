# START DOCKER CONTAINER

docker run --name takagi_agv \
	--gpus all \
	-dit \
  -p 40870:8888 \
	-v ~/investigation/SeniorThesisAGV/working:/workdir/working \
  -w /workdir/working \
	-e JUPYTER_TOKEN=555Stand \
	--cpuset-cpus=24-31 \
	takagi_agv:latest