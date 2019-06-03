import json


if __name__ == "__main__":

    set_names = ["test"]
    for set_name in set_names:
        input_file_path = f"preprocessed/multirc/multirc-{set_name}-processed-questions.jsonl"
        output_file_path = f"preprocessed/multirc/multirc-{set_name}-processed-questions-answerwise.jsonl"
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file.readlines():
                if line.strip():
                    instances_json = json.loads(line.strip())
                    paragraph_id = instances_json["paragraph_id"]
                    question_idx = instances_json["question_idx"]
                    premises = instances_json["premises"]
                    relevant_sentence_idxs = instances_json["relevant_sentence_idxs"]
                    hypotheses = instances_json["hypotheses"]
                    entailments = instances_json["entailments"]
                    for answer_idx, (hypothesis, entailment) in enumerate(zip(hypotheses, entailments)):
                        instance_json = {"premises": premises,
                                         "relevant_sentence_idxs": relevant_sentence_idxs,
                                         "hypothesis": hypothesis,
                                         "entailment": entailment,
                                         "answer_idx": answer_idx,
                                         "paragraph_id": paragraph_id,
                                         "question_idx": question_idx}
                        output_file.write(json.dumps(instance_json)+"\n")
