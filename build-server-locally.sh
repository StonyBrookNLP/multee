#!/bin/bash

set -e

echo ------------------------------
echo Building server
echo ------------------------------
echo

set -x
docker build -t multee-server -f Dockerfile-server .
set +x

echo
echo ------------------------------
echo Done
echo ------------------------------
echo

echo "To run server in Docker, do this:"
echo
echo "    docker run -p 8123:8123 --entrypoint /app/start_server.sh -t multee-server"
echo
echo "And visit http://localhost:8123"
echo

