#!/bin/bash

PYTHONUNBUFFERED=yes PYTHONPATH=. python \
  server/server.py \
  --archive-path trained_models/final_multee_glove_openbookqa.tar.gz \
  --predictor single_correct_mcq_entailment \
  --include-package lib \
  --port 8123 \
  --premise-retriever aws-es \
  --static-dir server/static_files/
