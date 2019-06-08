#!/usr/bin/env bash

set -e
set -x

export DATA_DIR_ROOT=../data/raw
export KNOWLEDGE_DIR_ROOT=../data/raw/OpenBookQA-V1-Sep2018/Data/knowledge
export OPENBOOKQA_DIR=${DATA_DIR_ROOT}/OpenBookQA-V1-Sep2018
export OUTPUT_DIR=${OPENBOOKQA_DIR}/Data/Main/ranked_knowledge

function retrieve() {
  set -e
  set -x
  mkdir -p "${OUTPUT_DIR}/$1" && \
  PYTHONPATH=. python obqa/data/retrieval/knowledge/rank_knowledge_for_mc_qa.py \
    -o "${OUTPUT_DIR}/$1" \
    -i "${OPENBOOKQA_DIR}/Data/Main/full.jsonl" \
    -k "$2" \
    -n tfidf $3
}
export -f retrieve

parallel --halt now,fail=1 --line-buffer --colsep , 'retrieve' ::: <<EOF
openbook,${KNOWLEDGE_DIR_ROOT}/openbook.csv,--max_facts_per_choice 100 --limit_items 0
cn5wordnet,${KNOWLEDGE_DIR_ROOT}/CN5/cn5_wordnet.json,--max_facts_per_choice 100 --limit_items 0
EOF
