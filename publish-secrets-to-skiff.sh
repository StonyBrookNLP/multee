#!/bin/bash

set -e

echo -----------------------------------
echo Getting secrets from Vault
echo -----------------------------------
echo

# Vault secrets were once written like this:
#
#     vault write /secret/aristo/skiff-deployments/multee/env-vars \
#         AWS_ES_HOSTNAME=xxx \
#         AWS_ES_REGION=xxx \
#         AWS_ES_INDEX=xxx \
#         AWS_ES_DOCUMENT_TYPE=xxx \
#         AWS_ES_FIELD_NAME=xxx \
#         AWS_ACCESS_KEY_ID=xxx \
#         AWS_SECRET_ACCESS_KEY=xxx

export AWS_ES_HOSTNAME=$(       vault read -field=AWS_ES_HOSTNAME       secret/aristo/skiff-deployments/multee/env-vars)
export AWS_ES_REGION=$(         vault read -field=AWS_ES_REGION         secret/aristo/skiff-deployments/multee/env-vars)
export AWS_ES_INDEX=$(          vault read -field=AWS_ES_INDEX          secret/aristo/skiff-deployments/multee/env-vars)
export AWS_ES_DOCUMENT_TYPE=$(  vault read -field=AWS_ES_DOCUMENT_TYPE  secret/aristo/skiff-deployments/multee/env-vars) 
export AWS_ES_FIELD_NAME=$(     vault read -field=AWS_ES_FIELD_NAME     secret/aristo/skiff-deployments/multee/env-vars)
export AWS_ACCESS_KEY_ID=$(     vault read -field=AWS_ACCESS_KEY_ID     secret/aristo/skiff-deployments/multee/env-vars)
export AWS_SECRET_ACCESS_KEY=$( vault read -field=AWS_SECRET_ACCESS_KEY secret/aristo/skiff-deployments/multee/env-vars)

echo -----------------------------------
echo Deleting secret in Kubernetes
echo -----------------------------------
echo

kubectl \
  --context skiff-production \
  --namespace aristo-multee \
  --ignore-not-found=true \
  delete secret env-vars

echo
echo -----------------------------------
echo Writing new values to Kubernetes
echo -----------------------------------
echo

kubectl \
  --context skiff-production \
  --namespace aristo-multee \
  create secret generic env-vars \
  --from-literal="AWS_ES_HOSTNAME=$AWS_ES_HOSTNAME" \
  --from-literal="AWS_ES_REGION=$AWS_ES_REGION" \
  --from-literal="AWS_ES_INDEX=$AWS_ES_INDEX" \
  --from-literal="AWS_ES_DOCUMENT_TYPE=$AWS_ES_DOCUMENT_TYPE" \
  --from-literal="AWS_ES_FIELD_NAME=$AWS_ES_FIELD_NAME" \
  --from-literal="AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" \
  --from-literal="AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"

echo
echo If we got this far, then everything probably succeeded.
