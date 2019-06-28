#!/bin/bash

# This starts a Multee server using the hard-coded premise retriever. It uses a
# short hard-coded list of sentences as premises when evaluating hypotheses.
# 
# See code in server.py and create a premise retriever that does real retrieval.

set -e


export PYTHONUNBUFFERED=yes
export PYTHONPATH=.

python \
  server/server.py \
  --archive-path trained_models/final_multee_glove_openbookqa.tar.gz \
  --predictor single_correct_mcq_entailment \
  --include-package lib \
  --port 8123 \
  --premise-retriever hard-coded \
  --static-dir server/static_files/
