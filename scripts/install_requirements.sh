#!/bin/bash

set -e
set -x

pip install -r requirements.txt

python -m spacy download en
python -m spacy download en_core_web_lg
