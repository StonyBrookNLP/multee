#!/bin/bash

# This starts a Multee web server.
#
# It is expected that you set the env var MULTEE_PREMISE_RETRIEVER (and other
# variables, like AWS_*), and that you have the trained_models directory
# populated with the necessary files.

set -e

export PYTHONUNBUFFERED=yes
export PYTHONPATH=.

python \
  server/server.py \
  --archive-path trained_models/final_multee_glove_openbookqa.tar.gz \
  --predictor single_correct_mcq_entailment \
  --aristo-predictor aristo_question \
  --include-package lib \
  --port 8123 \
  --static-dir server/static_files/

