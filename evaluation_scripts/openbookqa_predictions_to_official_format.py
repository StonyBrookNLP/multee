"""
Converts allennlp predictions from single-choice-mcq-entailment predictor to official openbookqa format for Leaderboard.
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
    index2label = ["A", "B", "C", "D"]
    with open(args.allennlp_predictions_file, 'r') as allennlp_predictions_file:
        for line in tqdm(allennlp_predictions_file.readlines()):
            if not line.strip():
                continue
            prediction_json = json.loads(line.strip())
            question_id = prediction_json["input"]["question_id"]
            predicted_answer_index = prediction_json["predicted_answer_index"]
            predicted_answer_label = index2label[predicted_answer_index]
            official_prediction_instances.append(f"{question_id},{predicted_answer_label}")

    with open(args.official_predictions_file, 'w') as file:
        file.write("\n".join(official_prediction_instances))
