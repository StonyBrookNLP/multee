#!/bin/bash

# This makes predictions on Aristo questions with Multee. It uses a hard-coded
# premise retriever, intended for testing purposes. If this script exits with
# code 0, then nothing failed and the expected predictions were made.

set -e

echo --------------------------------
echo Predicting with Multee
echo --------------------------------
echo

set -x
MULTEE_PREMISE_RETRIEVER=hard-coded ./predict-aristo-questions.sh tests/aristo-predictions/questions.jsonl /tmp/multee-predictions.jsonl
set +x


echo
echo --------------------------------
echo Checking predictions
echo --------------------------------
echo

set -x
diff -u tests/aristo-predictions/expected-predictions.jsonl /tmp/multee-predictions.jsonl
set +x

echo If we got this far, then things are good!


