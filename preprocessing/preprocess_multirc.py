"""
Reads raw multirc data dumps in following jsonl format:

    {
        "paragraph_id": ...,
        "question_idx": ...,
        "raw_question": "..."
        "premises": ["...", "...", "..."]
        "relevant_sentence_idxs": [0, 2, 5]
        "hypotheses": [ "...", "...", "..."]
        "entailments": [true/false, true/false, ...]
        "multisent": true/false
    }
    ...
"""

import json
import math
import re
import os

from ftfy import fix_text
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy

from lib.question_answer_to_explicit_hypothesis import question_answer_to_explicit_hypothesis
from lib.question_to_implicit_hypothesis import question_to_implicit_hypothesis
from lib.coref_replace import coref_replace

nlp = spacy.load('en')


if __name__ == "__main__":

    set_names = ["train", "dev"]
    for set_name in set_names:
        print(f"Processing {set_name} set ...")
        dataset_path = f"data/raw/multirc_1.0/multirc_1.0_{set_name}.json"
        question_dataset_path = f"data/preprocessed/multirc/multirc-{set_name}-processed-questions.jsonl"
        
        target_dir = os.path.dirname(question_dataset_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        with open(question_dataset_path, "w") as write_file:
            with open(dataset_path, "r") as read_file:
                dataset = json.load(read_file)
                paragraphs = dataset["data"]

                for paragraph in tqdm(paragraphs):
                    paragraph_id = paragraph["id"]

                    paragraph = paragraph["paragraph"]
                    paragraph_text = fix_text(paragraph["text"])
                    questions = paragraph["questions"]

                    original_sentences_count = len(re.findall(r'<b>Sent \d+: *<\/b>(.+?)<br>',
                                                              paragraph_text.strip()))
                    processed_paragraph_sentences = coref_replace(paragraph_text)

                    # Sentence boundaries must be maintained.
                    assert len(processed_paragraph_sentences) == original_sentences_count

                    for question in questions:
                        question_idx = question["idx"]
                        question_text = question["question"]
                        answers = question["answers"]
                        multisent = question["multisent"]
                        relevant_sentence_idxs = question["sentences_used"]

                        implicit_hypothesis = question_to_implicit_hypothesis(question_text)

                        explicit_hypotheses = []
                        entailments = []
                        for answer in answers:
                            answer_text = answer["text"]
                            explicit_hypothesis = question_answer_to_explicit_hypothesis(question_text,
                                                                                         answer_text,
                                                                                         implicit_hypothesis,
                                                                                         True)
                            entailment = answer["isAnswer"]
                            explicit_hypotheses.append(explicit_hypothesis)
                            entailments.append(entailment)

                        data_dict = {"paragraph_id": paragraph_id,
                                     "premises": processed_paragraph_sentences,
                                     "raw_question": question_text,
                                     "relevant_sentence_idxs": relevant_sentence_idxs,
                                     "question_idx": question_idx,
                                     "multisent": multisent,
                                     "hypotheses": explicit_hypotheses,
                                     "entailments": entailments}

                        write_file.write(json.dumps(data_dict) + "\n")
