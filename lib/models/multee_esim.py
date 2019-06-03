from typing import Dict, Optional, List, Any
import copy

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax

from lib.nn.util import paragraph2sentences_tensor, sentencewise_scores2paragraph_tokenwise_scores, sentences2paragraph_tensor, masked_divide
from lib.modules import EsimComparator, CoverageLoss
from lib.modules.esim_comparator import EsimComparatorLayer1, EsimComparatorLayer2, EsimComparatorLayer3Plus


@Model.register("multee_esim")
class MulteeEsim(Model):
    """
    Multee instantiated from ESIM. Takes 1 hypothesis at a time.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 final_feedforward: FeedForward,
                 coverage_loss: CoverageLoss = None,
                 contextualize_pair_comparators: bool = False,
                 pair_context_encoder: Seq2SeqEncoder = None,
                 pair_feedforward: FeedForward = None,
                 optimize_coverage_for: List = ["entailment", "neutral"],
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self._label2idx = self.vocab.get_token_to_index_vocabulary('labels')

        self._text_field_embedder = text_field_embedder

        self._entailment_comparator_layer_1 = EsimComparatorLayer1(encoder, dropout)
        self._entailment_comparator_layer_2 = EsimComparatorLayer2(similarity_function)

        self._td_entailment_comparator_layer_1 = TimeDistributed(self._entailment_comparator_layer_1)
        self._td_entailment_comparator_layer_2 = TimeDistributed(self._entailment_comparator_layer_2)

        self._entailment_comparator_layer_3plus_local = EsimComparatorLayer3Plus(projection_feedforward, inference_encoder,
                                                                                 output_feedforward, dropout)
        self._td_entailment_comparator_layer_3plus_local = TimeDistributed(self._entailment_comparator_layer_3plus_local)

        self._entailment_comparator_layer_3plus_global = copy.deepcopy(self._entailment_comparator_layer_3plus_local)

        self._contextualize_pair_comparators = contextualize_pair_comparators

        if not self._contextualize_pair_comparators:
            self._output_logit = output_logit
            self._td_output_logit = TimeDistributed(self._output_logit)

        self._final_feedforward = final_feedforward
        self._td_final_feedforward = TimeDistributed(final_feedforward)

        linear = torch.nn.Linear(2*self._entailment_comparator_layer_3plus_local.get_output_dim(),
                                 self._final_feedforward.get_input_dim())
        self._local_global_projection = torch.nn.Sequential(linear, torch.nn.ReLU())

        if self._contextualize_pair_comparators:
            self._pair_context_encoder = pair_context_encoder
            self._td_pair_feedforward = TimeDistributed(pair_feedforward)

        self._coverage_loss = coverage_loss

        # Do not apply initializer. If you do, make sure it doesn't reinitialize transferred parameters.

    @overrides
    def forward(self,  # type: ignore
                premises: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                paragraph: Dict[str, torch.LongTensor] = None,
                relevance_presence_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        premises_embeddings = self._text_field_embedder(premises, num_wrapping_dims=1)
        # premises_embeddings: (batch_size X num_sentences X seq_len X emb_dim)

        premises_mask = get_text_field_mask(premises, num_wrapping_dims=1).float()
        # premises_mask: (batch_size X num_sentences X seq_len )

        paragraph_embeddings = self._text_field_embedder(paragraph)
        # paragraph_embeddings: (batch_size X seq_len X emb_dim)

        paragraph_mask = get_text_field_mask(paragraph).float()
        # paragraph_mask: (batch_size X seq_len )

        hypothesis_embeddings = self._text_field_embedder(hypothesis)
        # hypothesis_embeddings: (batch_size X seq_len X emb_dim)

        hypothesis_mask = get_text_field_mask(hypothesis).float()
        # hypothesis_mask: (batch_size X seq_len)

        premises_presence_mask = premises_mask[:, :, 0]
        # premises_presence_mask: (batch_size X num_sentences)

        sentence_lengths = premises_mask.sum(dim=-1)
        # sentence_lengths: (batch_size X num_sentences)

        premises_count = premises_embeddings.shape[1]

        encoded_paragraph = self._entailment_comparator_layer_1(paragraph_embeddings, paragraph_mask)
        encoded_hypothesis = self._entailment_comparator_layer_1(hypothesis_embeddings, hypothesis_mask)

        hypotheses_mask = hypothesis_mask.unsqueeze(1)*premises_presence_mask.unsqueeze(-1)

        encoded_premises = paragraph2sentences_tensor(encoded_paragraph, sentence_lengths)
        encoded_hypotheses = encoded_hypothesis.unsqueeze(1)*premises_presence_mask.unsqueeze(-1).unsqueeze(-1)
        similarity_matrices = self._td_entailment_comparator_layer_2(encoded_premises, encoded_hypotheses)
        similarity_matrix = sentences2paragraph_tensor(similarity_matrices, premises_mask)

        # local entailment comparators:
        local_entailment_comparators = self._td_entailment_comparator_layer_3plus_local(encoded_premises,
                                                                                        encoded_hypotheses,
                                                                                        similarity_matrices,
                                                                                        premises_mask,
                                                                                        hypotheses_mask)

        # get local_entailment_scores:
        if self._contextualize_pair_comparators:
            contextualized_pair_comparators = self._pair_context_encoder(local_entailment_comparators, premises_presence_mask)
            pair_comparators = torch.cat([contextualized_pair_comparators, local_entailment_comparators], dim=-1)
            pair_entailment_scores = self._td_pair_feedforward(pair_comparators).squeeze(-1)*premises_presence_mask
        else:
            pair_comparators = local_entailment_comparators
            pair_logits = self._td_output_logit(pair_comparators)
            pair_probs = pair_logits.softmax(dim=-1)
            pair_entailment_scores = pair_probs[:, :, self._label2idx["entailment"]]

        premises_attention = masked_softmax(pair_entailment_scores, premises_presence_mask, dim=1)

        paragraph_tokenwise_probs = sentencewise_scores2paragraph_tokenwise_scores(premises_attention, premises_mask)

        # global entailment comparator:
        global_entailment_comparator = self._entailment_comparator_layer_3plus_global(encoded_paragraph,
                                                                                      encoded_hypothesis,
                                                                                      similarity_matrix,
                                                                                      paragraph_mask,
                                                                                      hypothesis_mask,
                                                                                      paragraph_tokenwise_probs)

        # weighted local entailment comparator:
        weighted_local_entailment_comparator = torch.sum(premises_attention.unsqueeze(-1)*local_entailment_comparators, dim=1)

        # Final Aggregated Representation:
        final_representation = torch.cat([weighted_local_entailment_comparator, global_entailment_comparator], dim=-1)
        final_representation = self._local_global_projection(final_representation)

        label_logits = self._final_feedforward(final_representation)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits,
                       "premises_attention": premises_attention}

        if relevance_presence_mask is not None:
            coverage_loss = self._coverage_loss(pair_entailment_scores, premises_presence_mask, relevance_presence_mask)
            output_dict["coverage_loss"] = coverage_loss

        return output_dict
