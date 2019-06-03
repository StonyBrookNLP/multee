"""
Reads the data and knowledge sources from OBQA's official retrieval code and dumps it in the following jsonl format:

    {
        "question_idx":
        "raw_question": "..."
        "premises": ["...", "...", "..."]
        "hypotheses": [ "...", "...", "..."]
        "entailments": [true/false, true/false, ...]
    }
    ...
"""

import os
import sys
import json

from typing import Iterable, Dict
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, "allennlp"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, "OpenBookQA"))

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Instance
from allennlp.common.util import import_submodules

from lib.question_answer_to_explicit_hypothesis import question_answer_to_explicit_hypothesis
from lib.question_to_implicit_hypothesis import question_to_implicit_hypothesis

import_submodules("obqa.data.dataset_readers")

if __name__ == "__main__":

    use_knowledge_sources = ["openbook", "cn5wordnet"]
    max_facts_per_argument = 5

    reader_params = Params({"dataset_reader": {
                              "type": "arc-multi-choice-w-facts-txt-json-multi-source",
                              "question_value_type": "question",
                              "token_indexers": {
                                  "tokens": {
                                      "type": "single_id",
                                      "lowercase_tokens": True
                                  }
                              },
                              "external_knowledge": {
                                  "sources": [
                                       {
                                        "type": "flexible-json",
                                          "name": "cn5wordnet",
                                          "use_cache": True,
                                          "dataset_to_know_json_file": {"any": "data/raw/OpenBookQA-V1-Sep2018/Data/Main/ranked_knowledge/cn5wordnet/knowledge.json"},
                                          "dataset_to_know_rank_file": {"any": "data/raw/OpenBookQA-V1-Sep2018/Data/Main/ranked_knowledge/cn5wordnet/full.jsonl.ranking.json"},
                                          "rank_reader_type": "flat-q-ch-values-v1",
                                          "max_facts_per_argument": max_facts_per_argument
                                       },
                                       {
                                        "type": "flexible-json",
                                          "name": "openbook",
                                          "use_cache": True,
                                          "dataset_to_know_json_file": {"any": "data/raw/OpenBookQA-V1-Sep2018/Data/Main/ranked_knowledge/openbook/knowledge.json"},
                                          "dataset_to_know_rank_file": {"any": "data/raw/OpenBookQA-V1-Sep2018/Data/Main/ranked_knowledge/openbook/full.jsonl.ranking.json"},
                                          "rank_reader_type": "flat-q-ch-values-v1",
                                          "max_facts_per_argument": max_facts_per_argument
                                       },
                                   ],
                                "sources_use": [
                                    "cn5wordnet" in use_knowledge_sources,
                                    "openbook" in use_knowledge_sources
                                ]
                              }
                            }
                           })

    set_names = ["train", "dev", "test"]
    dataset_reader = DatasetReader.from_params(reader_params.pop('dataset_reader'))

    knowledge_sources_str = "_".join(use_knowledge_sources)

    for set_name in set_names:
        print(f"Working on {set_name}...")

        original_data_path = f"data/raw/OpenBookQA-V1-Sep2018/Data/Main/{set_name}.jsonl"
        preprocessed_file = f"data/preprocessed/openbookqa/openbookqa-{set_name}-processed-questions.jsonl"

        target_dir = os.path.dirname(preprocessed_file)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        instances = dataset_reader.read(original_data_path)

        with open(preprocessed_file, "w") as write_file:
            for instance in tqdm(instances):
                metadata = instance.fields['metadata'].metadata

                question_id = metadata['id']
                question_text = metadata['question_text']
                answer_choices = metadata['choice_text_list']
                answerkey_idx = metadata['label_gold']
                premises = metadata['facts_text_list']

                implicit_hypothesis = question_to_implicit_hypothesis(question_text)

                explicit_hypotheses = []
                entailments = []
                for answer_idx, answer_text in enumerate(answer_choices):
                    explicit_hypothesis = question_answer_to_explicit_hypothesis(question_text,
                                                                                 answer_text,
                                                                                 implicit_hypothesis,
                                                                                 False)
                    entailment = (answer_idx == answerkey_idx)
                    entailments.append(entailment)
                    explicit_hypotheses.append(explicit_hypothesis)

                instance = {}
                instance["premises"] = premises
                instance["raw_question"] = question_text
                instance["question_id"] = question_id
                instance["hypotheses"] = explicit_hypotheses
                instance["entailments"] = entailments

                write_file.write(json.dumps(instance)+"\n")
