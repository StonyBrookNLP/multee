#!/bin/bash

# Use this script to run a Multee prediction on Aristo questions.
#
# It is expected that you set the env var MULTEE_PREMISE_RETRIEVER (and other
# variables, like AWS_*), and that you have the trained_models directory
# populated with the necessary files.

set -e

INFILE=${1?"Input file name (questions.jsonl) required"}
OUTFILE=${2?"Output file name (predictions.jsonl) required"}

export PYTHONUNBUFFERED=yes
export PYTHONPATH=.

allennlp predict \
  --predictor aristo_question \
  --include-package lib \
  trained_models/final_multee_glove_openbookqa.tar.gz \
  --output-file $OUTFILE \
  $INFILE

