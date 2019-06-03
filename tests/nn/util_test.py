# pylint: disable=invalid-name,no-self-use,too-many-public-methods,not-callable
from typing import Dict

import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.fields import Field, ListField, TextField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.nn.util import get_text_field_mask
from allennlp.common.params import Params
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from lib.nn.util import sentences2paragraph_tensor, paragraph2sentences_tensor

class TestSentenceParagraphConversions(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        token_indexers = {"tokens": SingleIdTokenIndexer(namespace="tokens")}
        tokenizer = WordTokenizer()

        list_of_sentences = [
                ["words1 words2 words3", "words1 words2"],
                ["words1", "words1 words2"]
        ]

        paragraphs = [
                ["words1 words2 words3 words1 words2"],
                ["words1 words1 words2"]
        ]

        instances = []
        for sentences, paragraph in zip(list_of_sentences, paragraphs):
            sentences_tokens = [tokenizer.tokenize(sentence) for sentence in sentences]
            sentences_text_fields = [TextField(sentence_tokens, token_indexers)
                                     for sentence_tokens in sentences_tokens]
            sentences_field = ListField(sentences_text_fields)

            fields: Dict[str, Field] = {}
            fields['sentences'] = sentences_field
            paragraph_tokens = [token for sentence_tokens in sentences_tokens for token in sentence_tokens]
            paragraph_text_field = TextField(paragraph_tokens, token_indexers)
            fields['paragraph'] = paragraph_text_field
            instances.append(Instance(fields))

        vocab = Vocabulary.from_instances(instances)
        batch = Batch(instances)
        batch.index_instances(vocab)

        tensor_dict = batch.as_tensor_dict()

        sentences = tensor_dict["sentences"]
        paragraph = tensor_dict["paragraph"]

        text_field_embedder_params = Params({
                "token_embedders": {
                        "tokens": {
                                "type": "embedding",
                                "embedding_dim": 3
                                },
                        }
                })
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab,
                                                                 params=text_field_embedder_params)

        self.embedded_sentences = text_field_embedder(sentences, num_wrapping_dims=1)
        self.sentences_mask = get_text_field_mask(sentences, num_wrapping_dims=1).float()
        self.sentences_lengths = self.sentences_mask.sum(dim=-1)

        self.embedded_paragraph = text_field_embedder(paragraph)
        self.paragraph_mask = get_text_field_mask(paragraph).float()

    def test_sentences2paragraph_tensor(self):
        embedded_paragraph = sentences2paragraph_tensor(self.embedded_sentences,
                                                        self.sentences_mask)
        assert torch.all(embedded_paragraph*self.paragraph_mask.unsqueeze(-1) == self.embedded_paragraph*self.paragraph_mask.unsqueeze(-1))

    def test_paragraph2sentences_tensor(self):
        embedded_sentences = paragraph2sentences_tensor(self.embedded_paragraph, self.sentences_lengths)
        assert torch.all(self.embedded_sentences*self.sentences_mask.unsqueeze(-1) == embedded_sentences*self.sentences_mask.unsqueeze(-1))

    def test_sentencewise_scores2paragraph_tokenwise_scores(self):
        # yet to be tested ...
        pass
