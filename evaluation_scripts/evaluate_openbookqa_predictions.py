"""
Takes raw-data-file and official (leaderboard) predictions format and prints performance for openbookqa.
"""

import json
import argparse

def eval(labels_file, predictions_file):

    questionid2predicted_label = {}
    with open(predictions_file, "r") as file:
        for line in file.readlines():
            if not line.strip():
                continue
            question_id, predicted_label = line.strip().split(",")
            questionid2predicted_label[question_id] = predicted_label

    correct_count = 0
    total_count = 0
    with open(labels_file, "r") as file:
        for line in file.readlines():
            if not line.strip():
                continue
            gold_instance = json.loads(line.strip())
            question_id = gold_instance["id"]
            correct_answer_label = gold_instance["answerKey"]
            if questionid2predicted_label[question_id] == correct_answer_label:
                correct_count += 1
            total_count += 1
    accuracy = float(correct_count)/total_count
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data_file', type=str, help='raw data file')
    parser.add_argument('official_predictions_file', type=str, help='official predictions file')
    args = parser.parse_args()
    eval(args.raw_data_file, args.official_predictions_file)
