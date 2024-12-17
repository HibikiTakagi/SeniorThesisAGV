# START DOCKER CONTAINER

docker run --name takagi_agv \
	--gpus all \
	-dit \
	-v ~/investigation/SeniorThesisAGV/working:/working \
  -w /working \
	-e JUPYTER_TOKEN=315Stand \
	--cpuset-cpus=24-31 \
	takagi_agv:latest