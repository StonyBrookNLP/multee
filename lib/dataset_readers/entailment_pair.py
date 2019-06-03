import logging
from typing import Dict
from overrides import overrides
import json

import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("entailment_pair")
class EntailmentPairReader(DatasetReader):
    """
    It's same reader as snli reader, except here we allow sentence1 or premise,
    sentence2 or hypothesis keys, and label or gold_label keys.
    """
    def __init__(self,
                 max_tokens: int = 200,
                 max_tuples: int = 300,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)

        self._max_tokens = max_tokens
        self._max_tuples = max_tuples
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as entailment_file:
            logger.info("Reading entailment instances from jsonl dataset at: %s", file_path)
            for line in entailment_file:
                if line.strip():
                    instance_json = json.loads(line.strip())
                    premise = instance_json.get("sentence1", None) or instance_json.get("premise", None)
                    hypothesis = instance_json.get("sentence2", None) or instance_json.get("hypothesis", None)
                    label = instance_json.get("gold_label", None) or instance_json.get("label", None) # entails or neutral
                    if label == '-':
                        # These were cases where the annotators disagreed; we'll just skip them.
                        # It's like 800 out of 500k examples in the training data.
                        continue
                    if label in ["entails", "entailment"]:
                        label = "entailment"
                    yield self.text_to_instance(premise, hypothesis, label)

    @overrides
    def text_to_instance(self, # pylint: disable=arguments-differ
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        premise_tokens = [Token(token.text)
                          for token in self._tokenizer.tokenize(premise)[-self._max_tokens:]]
        hypothesis_tokens = [Token(token.text)
                             for token in self._tokenizer.tokenize(hypothesis)[-self._max_tokens:]]

        fields['premise'] = TextField(premise_tokens, self._token_indexers)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)

        if label:
            fields['label'] = LabelField(label)

        # metadata = {"premise_tokens": [x.text for x in premise_tokens],
        #             "hypothesis_tokens": [x.text for x in hypothesis_tokens]}
        # fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

