"""
Converts allennlp predictions from multiple-choice-mcq-entailment predictor to official multirc format for Leaderboard.
"""

import json
import argparse
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('allennlp_predictions_file', type=str, help='allennlp predictions file')
    parser.add_argument('official_predictions_file', type=str, help='official predictions file')
    args = parser.parse_args()

    official_prediction_instances = []
    with open(args.allennlp_predictions_file, 'r') as allennlp_predictions_file:
        for line in tqdm(allennlp_predictions_file.readlines()):
            if not line.strip():
                continue
            prediction_json = json.loads(line.strip())
            paragraph_id = prediction_json["input"]["paragraph_id"]
            question_idx = prediction_json["input"]["question_idx"]
            answer_choices_count = len(prediction_json["input"]["hypotheses"])
            predicted_answer_indices = prediction_json["predicted_answer_indices"]
            scores = [int(index in predicted_answer_indices) for index in range(0, answer_choices_count)]
            official_prediction_instances.append({"qid": question_idx,
                                                  "pid": paragraph_id,
                                                  "scores": scores})

    with open(args.official_predictions_file, 'w') as file:
        json.dump(official_prediction_instances, file)
