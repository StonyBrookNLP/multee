from typing import Dict, Optional, List
import copy
import re

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from lib.modules import CoverageLoss
from lib.nn.util import unbind_tensor_dict
from lib.models.multee_esim import MulteeEsim


@Model.register("multiple_correct_mcq_multee_esim")
class MultipleCorrectMcqMulteeEsim(MulteeEsim):

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
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         encoder=encoder,
                         similarity_function=similarity_function,
                         projection_feedforward=projection_feedforward,
                         inference_encoder=inference_encoder,
                         output_feedforward=output_feedforward,
                         output_logit=output_logit,
                         final_feedforward=final_feedforward,
                         coverage_loss=coverage_loss,
                         contextualize_pair_comparators=contextualize_pair_comparators,
                         pair_context_encoder=pair_context_encoder,
                         pair_feedforward=pair_feedforward,
                         dropout=dropout,
                         initializer=initializer,
                         regularizer=regularizer)
        self._ignore_index = -1
        self._answer_loss = torch.nn.CrossEntropyLoss(ignore_index=self._ignore_index)
        self._coverage_loss = coverage_loss

        self._accuracy = CategoricalAccuracy()
        self._entailment_f1 = F1Measure(self._label2idx["entailment"])

    @overrides
    def forward(self,  # type: ignore
                premises: Dict[str, torch.LongTensor],
                hypotheses: Dict[str, torch.LongTensor],
                paragraph: Dict[str, torch.LongTensor],
                answer_correctness_mask: torch.IntTensor = None,
                relevance_presence_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        hypothesis_list = unbind_tensor_dict(hypotheses, dim=1)

        label_logits = []
        premises_attentions = []
        coverage_losses = []
        for hypothesis in hypothesis_list:
            output_dict = super().forward(premises=premises, hypothesis=hypothesis,
            	                          paragraph=paragraph, relevance_presence_mask=relevance_presence_mask)
            individual_logit = output_dict["label_logits"]
            label_logits.append(individual_logit)

            if relevance_presence_mask is not None:
                premises_attention = output_dict["premises_attention"]
                premises_attentions.append(premises_attention)
                coverage_loss = output_dict["coverage_loss"]
                coverage_losses.append(coverage_loss)

        label_logits = torch.stack(label_logits, dim=1)
        if relevance_presence_mask is not None:
            premises_attentions = torch.stack(premises_attentions, dim=1)
            coverage_losses = torch.stack(coverage_losses, dim=0)

        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
        output_dict = {"label_logits": label_logits[:, :, self._label2idx["entailment"]],
                       "label_probs": label_probs[:, :, self._label2idx["entailment"]]}
        if relevance_presence_mask is not None:
            output_dict["premises_attentions"] = premises_attentions

        if answer_correctness_mask is not None:
            label = ((answer_correctness_mask == 1).long()*self._label2idx["entailment"]
                     + (answer_correctness_mask == 0).long()*self._label2idx["neutral"]
                     + (answer_correctness_mask == -1).long()*self._ignore_index)
            loss = self._answer_loss(label_logits.reshape((-1, label_logits.shape[-1])), label.reshape((-1)))

            # coverage loss
            if relevance_presence_mask is not None:
                loss += coverage_losses.mean()
            output_dict["loss"] = loss

            mask = answer_correctness_mask != -1
            self._accuracy(label_logits, label, mask)
            self._entailment_f1(label_logits, label, mask)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy_metric = self._accuracy.get_metric(reset)
        entailment_f1_metric = self._entailment_f1.get_metric(reset)
        return {'_accuracy': accuracy_metric,
                '_entailment_prec': entailment_f1_metric[0],
                '_entailment_rec': entailment_f1_metric[1],
                'entailment_f1': entailment_f1_metric[2]}
