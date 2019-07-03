#!/bin/bash

# Use this script to run a Multee prediction on Aristo questions when you're on the AI2 network.

set -e

#INFILE=${1?"Input file name (questions.jsonl) required"}
#OUTFILE=${2?"Output file name (predictions.jsonl) required"}
#
export MULTEE_PREMISE_RETRIEVER=aws-es

# Unset the AWS_PROFILE env var, so it doesn't conflict with the
# AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars we're setting below.
unset AWS_PROFILE

export AWS_ES_HOSTNAME=$(       vault read -field=AWS_ES_HOSTNAME       /secret/aristo/multee-at-ai2/env-vars)
export AWS_ES_REGION=$(         vault read -field=AWS_ES_REGION         /secret/aristo/multee-at-ai2/env-vars)
export AWS_ES_INDEX=$(          vault read -field=AWS_ES_INDEX          /secret/aristo/multee-at-ai2/env-vars)
export AWS_ES_DOCUMENT_TYPE=$(  vault read -field=AWS_ES_DOCUMENT_TYPE  /secret/aristo/multee-at-ai2/env-vars)
export AWS_ES_FIELD_NAME=$(     vault read -field=AWS_ES_FIELD_NAME     /secret/aristo/multee-at-ai2/env-vars)
export AWS_ACCESS_KEY_ID=$(     vault read -field=AWS_ACCESS_KEY_ID     /secret/aristo/multee-at-ai2/env-vars)
export AWS_SECRET_ACCESS_KEY=$( vault read -field=AWS_SECRET_ACCESS_KEY /secret/aristo/multee-at-ai2/env-vars)

if [ "$AWS_ES_HOSTNAME" == "" ]; then
   echo "Missing env var AWS_ES_HOSTNAME. Perthaps you're not logged in to vault?"
   exit 1
fi

./predict-aristo-questions.sh "$@"
