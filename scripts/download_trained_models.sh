#!/bin/bash

set -e
set -x

mkdir -p trained_models/
cd trained_models/

# Download NLI glove model for MultiRC
ESIM_GLOVE_SNLI_MULTINLI_FOR_MULTIRC_URL=https://aristo-public-release.s3-us-west-2.amazonaws.com/Multee/models/final_esim_glove_snli_multinli_for_multirc_3Jun19.tar.gz
wget $ESIM_GLOVE_SNLI_MULTINLI_FOR_MULTIRC_URL
mv final_esim_glove_snli_multinli_for_multirc_3Jun19.tar.gz final_esim_glove_snli_multinli_for_multirc.tar.gz

# Download NLI glove model for OpenBookQA
ESIM_GLOVE_SNLI_MULTINLI_FOR_OPENBOOKQA_URL=https://aristo-public-release.s3-us-west-2.amazonaws.com/Multee/models/final_esim_glove_snli_multinli_for_openbookqa_3Jun19.tar.gz
wget $ESIM_GLOVE_SNLI_MULTINLI_FOR_OPENBOOKQA_URL
mv final_esim_glove_snli_multinli_for_openbookqa_3Jun19.tar.gz final_esim_glove_snli_multinli_for_openbookqa.tar.gz

# Download Multee glove model for OpenBookQA
MULTEE_GLOVE_OPENBOOKQA_URL=https://aristo-public-release.s3-us-west-2.amazonaws.com/Multee/models/final_multee_glove_multirc_3Jun19.tar.gz
wget $MULTEE_GLOVE_OPENBOOKQA_URL
mv final_multee_glove_multirc_3Jun19.tar.gz final_multee_glove_multirc.tar.gz

# Download Multee glove model for MultiRC
MULTEE_GLOVE_MULTIRC_URL=https://aristo-public-release.s3-us-west-2.amazonaws.com/Multee/models/final_multee_glove_openbookqa_3Jun19.tar.gz
wget $MULTEE_GLOVE_MULTIRC_URL
mv final_multee_glove_openbookqa_3Jun19.tar.gz final_multee_glove_openbookqa.tar.gz
