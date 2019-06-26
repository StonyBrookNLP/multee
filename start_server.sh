#!/bin/bash


# This assumes that we're in Multee's virtualenv, and that the .tar.gz model
# file exists.
#
# (To obtain this model file, run scripts/download_trained_models.sh)

PYTHONUNBUFFERED=yes PYTHONPATH=. python \
  server/server.py \
  --archive-path trained_models/final_multee_glove_openbookqa.tar.gz \
  --predictor single_correct_mcq_entailment \
  --include-package lib \
  --port 8123 \
  --static-dir server/static_files/
