#!/bin/bash

# Use this script to start a Multee server when you're on the AI2 network.

set -e


# Unset the AWS_PROFILE env var, so it doesn't conflict with the
# AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars we're setting below.
unset AWS_PROFILE

export AWS_ES_HOSTNAME=$(       vault read -field=AWS_ES_HOSTNAME       secret/aristo/skiff-deployments/multee/env-vars)
export AWS_ES_REGION=$(         vault read -field=AWS_ES_REGION         secret/aristo/skiff-deployments/multee/env-vars)
export AWS_ES_INDEX=$(          vault read -field=AWS_ES_INDEX          secret/aristo/skiff-deployments/multee/env-vars)
export AWS_ES_DOCUMENT_TYPE=$(  vault read -field=AWS_ES_DOCUMENT_TYPE  secret/aristo/skiff-deployments/multee/env-vars) 
export AWS_ES_FIELD_NAME=$(     vault read -field=AWS_ES_FIELD_NAME     secret/aristo/skiff-deployments/multee/env-vars)
export AWS_ACCESS_KEY_ID=$(     vault read -field=AWS_ACCESS_KEY_ID     secret/aristo/skiff-deployments/multee/env-vars)
export AWS_SECRET_ACCESS_KEY=$( vault read -field=AWS_SECRET_ACCESS_KEY secret/aristo/skiff-deployments/multee/env-vars)

export PYTHONUNBUFFERED=yes
export PYTHONPATH=.

python \
  server/server.py \
  --archive-path trained_models/final_multee_glove_openbookqa.tar.gz \
  --predictor single_correct_mcq_entailment \
  --include-package lib \
  --port 8123 \
  --premise-retriever aws-es \
  --static-dir server/static_files/
