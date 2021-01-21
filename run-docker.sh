#!/bin/bash
#run-docker.sh

# This guarantees that the docker image is up to date before
# running it.
#
docker build -t air-force-service .
docker run -d -p 5000:5000 air-force-service:latest
