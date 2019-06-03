"""
Takes raw-data-file and official (leaderboard) predictions format and prints performance for multirc.

Taken from https://github.com/CogComp/multirc/blob/master/multirc_materials/multirc-eval-v1.py
"""

import json
import argparse
from lib.multirc_measures import Measures

measures = Measures()

# the input to the `eval` function is the file which contains the binary predictions per question-id
def eval(labels_file, predictions_file):
    input = json.load(open(labels_file))
    output = json.load(open(predictions_file))
    output_map = dict([[a["pid"] + "==" + a["qid"], a["scores"]] for a in output])

    assert len(output_map) == len(output), "You probably have redundancies in your keys"

    [P1, R1, F1m] = measures.per_question_metrics(input["data"], output_map)
    print("Per question measures (i.e. precision-recall per question, then average) ")
    print("\tP: " + str(P1) + " - R: " + str(R1) + " - F1m: " + str(F1m))

    EM0 = measures.exact_match_metrics(input["data"], output_map, 0)
    EM1 = measures.exact_match_metrics(input["data"], output_map, 1)
    print("\tEM0: " + str(EM0))
    print("\tEM1: " + str(EM1))

    [P2, R2, F1a] = measures.per_dataset_metric(input["data"], output_map)

    print("Dataset-wide measures (i.e. precision-recall across all the candidate-answers in the dataset) ")
    print("\tP: " + str(P2) + " - R: " + str(R2) + " - F1a: " + str(F1a))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data_file', type=str, help='raw data file')
    parser.add_argument('official_predictions_file', type=str, help='official predictions file')
    args = parser.parse_args()
    eval(args.raw_data_file, args.official_predictions_file)
