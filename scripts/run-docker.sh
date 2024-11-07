#!/bin/bash

export containerName="toad$1_$USER"

docker run -d --gpus "device=1" --rm -it \
    --volume="/garage/mbenavent/TOAD/:/workspace:rw" \
    --volume="/mnt/md1/datasets/:/datasets:ro" \
	--volume="/mnt/nvme/mbenavent/OAD/:/nvme:rw" \
    --workdir="/workspace" \
	--name $containerName \
	--shm-size=32g \
	toad:latest bash