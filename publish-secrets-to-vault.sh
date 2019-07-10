#!/bin/bash

set -e

# To publish secrets to Vault, change the xxx values below and uncomment the lines.
#
# Don't commit this file with actual secrets!
# 
# vault write /secret/aristo/multee-on-skiff/env-vars \
#     AWS_ACCESS_KEY_ID=xxx \
#     AWS_ES_DOCUMENT_TYPE=xxx \
#     AWS_ES_FIELD_NAME=xxx \
#     AWS_ES_HOSTNAME=xxx \
#     AWS_ES_INDEX=xxx \
#     AWS_ES_REGION=xxx \
#     AWS_SECRET_ACCESS_KEY=xxx
# 
# vault write /secret/aristo/multee-at-ai2/env-vars \
#     AWS_ACCESS_KEY_ID=xxx \
#     AWS_ES_DOCUMENT_TYPE=xxx \
#     AWS_ES_FIELD_NAME=xxx \
#     AWS_ES_HOSTNAME=xxx \
#     AWS_ES_INDEX=xxx \
#     AWS_ES_REGION=xxx \
#     AWS_SECRET_ACCESS_KEY=xxx
# 
# vault write /secret/aristo/multee-on-beaker/env-vars \
#     AWS_ACCESS_KEY_ID=xxx \
#     AWS_ES_DOCUMENT_TYPE=xxx \
#     AWS_ES_FIELD_NAME=xxx \
#     AWS_ES_HOSTNAME=xxx \
#     AWS_ES_INDEX=xxx \
#     AWS_ES_REGION=xxx \
#     AWS_SECRET_ACCESS_KEY=xxx
