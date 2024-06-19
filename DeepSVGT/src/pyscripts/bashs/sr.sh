#!/bin/bash

# simlation
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/1phd:/workspace/1phd \
						--rm nvcr.io/nvidia/pytorch:22.07-py3-03 \
						python ./1phd/pyscripts/simrun.py \
