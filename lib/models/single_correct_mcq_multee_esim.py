from typing import Dict, Optional, List, Any
import copy
import re

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, InputVariationalDropout, Seq2VecEncoder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, replace_masked_values, get_mask_from_sequence_lengths
from allennlp.training.metrics import CategoricalAccuracy

from lib.modules import CoverageLoss
from lib.nn.util import unbind_tensor_dict
from lib.models.multee_esim import MulteeEsim


@Model.register("single_correct_mcq_multee_esim")
class SingleCorrectMcqMulteeEsim(MulteeEsim):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 final_feedforward: FeedForward,
                 coverage_loss: CoverageLoss,
                 similarity_function: SimilarityFunction = DotProductSimilarity(),
                 dropout: float = 0.5,
                 contextualize_pair_comparators: bool = False,
                 pair_context_encoder: Seq2SeqEncoder = None,
                 pair_feedforward: FeedForward = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
    	# Need to send it verbatim because otherwise FromParams doesn't work appropriately.
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         encoder=encoder,
                         similarity_function=similarity_function,
                         projection_feedforward=projection_feedforward,
                         inference_encoder=inference_encoder,
                         output_feedforward=output_feedforward,
                         output_logit=output_logit,
                         final_feedforward=final_feedforward,
                         contextualize_pair_comparators=contextualize_pair_comparators,
                         coverage_loss=coverage_loss,
                         pair_context_encoder=pair_context_encoder,
                         pair_feedforward=pair_feedforward,
                         dropout=dropout,
                         initializer=initializer,
                         regularizer=regularizer)
        self._answer_loss = torch.nn.CrossEntropyLoss()

        self._accuracy = CategoricalAccuracy()

    @overrides
    def forward(self,  # type: ignore
                premises: Dict[str, torch.LongTensor],
                hypotheses: Dict[str, torch.LongTensor],
                paragraph: Dict[str, torch.LongTensor],
                answer_index: torch.LongTensor = None,
                relevance_presence_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        hypothesis_list = unbind_tensor_dict(hypotheses, dim=1)

        label_logits = []
        premises_attentions = []
        coverage_losses = []
        for hypothesis in hypothesis_list:
            output_dict = super().forward(premises=premises, hypothesis=hypothesis, paragraph=paragraph)
            individual_logit = output_dict["label_logits"][:, self._label2idx["entailment"]] # only useful key
            label_logits.append(individual_logit)

            if relevance_presence_mask is not None:
                premises_attention = output_dict["premises_attention"]
                premises_attentions.append(premises_attention)
                coverage_loss = output_dict["coverage_loss"]
                coverage_losses.append(coverage_loss)

        label_logits = torch.stack(label_logits, dim=-1)
        if relevance_presence_mask is not None:
            premises_attentions = torch.stack(premises_attentions, dim=1)
            coverage_losses = torch.stack(coverage_losses, dim=0)

        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
        output_dict = {"label_logits": label_logits,
                       "label_probs": label_probs}

        if answer_index is not None:
            # answer_loss
            loss = self._answer_loss(label_logits, answer_index)
            # coverage loss
            if relevance_presence_mask is not None:
                loss += coverage_losses.mean()
            output_dict["loss"] = loss

            self._accuracy(label_logits, answer_index)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy_metric = self._accuracy.get_metric(reset)
        return {'accuracy': accuracy_metric}
