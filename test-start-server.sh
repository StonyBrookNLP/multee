#!/bin/bash

# This starts a Multee server with a hard-coded premise retriever, intended for
# testing purposes. To test, run this script, visit http://localhost:8123/ when
# the server starts, and try things out.

set -e

export MULTEE_PREMISE_RETRIEVER=hard-coded 

./start-server.sh
