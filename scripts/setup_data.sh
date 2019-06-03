#!/bin/bash

set -e
set -x

mkdir -p data/
mkdir -p data/raw
mkdir -p data/processed

# Download everything in raw data directory
cd data/raw/

# Download and Setup SNLI
SNLI_DATA_URL=https://nlp.stanford.edu/projects/snli/snli_1.0.zip
wget $SNLI_DATA_URL
unzip $(basename $SNLI_DATA_URL)
rm snli_1.0.zip


# Download and Setup MultiNLI
MULTINLI_DATA_URL=https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
wget $MULTINLI_DATA_URL
unzip $(basename $MULTINLI_DATA_URL)
rm multinli_1.0.zip


# Download and setup MultiRC
MULTIRC_DATA_URL=https://cogcomp.org/multirc/data/mutlirc-v2.zip
wget $MULTIRC_DATA_URL
unzip $(basename $MULTIRC_DATA_URL)
mv splitv2 multirc_1.0
mv multirc_1.0/dev_83-fixedIds.json multirc_1.0/multirc_1.0_dev.json 
mv multirc_1.0/train_456-fixedIds.json multirc_1.0/multirc_1.0_train.json 
rm mutlirc-v2.zip


# OpenBookoQA (Taken from OBQA repository)
OPENBOOKQA_DATA_URL="https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip"
wget $OPENBOOKQA_DATA_URL
unzip $(basename $OPENBOOKQA_DATA_URL)
rm OpenBookQA-V1-Sep2018.zip

# prepare data for retrieval
cd OpenBookQA-V1-Sep2018/Data

# knowledge
mkdir -p knowledge
cd knowledge
CN5_KNOW_DATA_URL="https://s3-us-west-2.amazonaws.com/ai2-website/data/knowledge_cn5.zip"
wget $CN5_KNOW_DATA_URL
unzip $(basename $CN5_KNOW_DATA_URL)
cd ..

cd Main
cat train.jsonl dev.jsonl test.jsonl > full.jsonl
# create knowledge file that the data retrieval can read
OPENBOOK_CSV=../knowledge/openbook.csv
echo "SCIENCE-FACT" > ${OPENBOOK_CSV}
cat openbook.txt >> ${OPENBOOK_CSV}
cd ..

cd Additional
cat train_complete.jsonl dev_complete.jsonl test_complete.jsonl > full_complete.jsonl
cd ../

cd ../..
