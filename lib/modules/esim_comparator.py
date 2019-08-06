from typing import Dict
import copy
import re

from overrides import overrides
import torch

from allennlp.modules import FeedForward, InputVariationalDropout
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, replace_masked_values
from allennlp.modules.matrix_attention.dot_product_matrix_attention import DotProductMatrixAttention

from lib.nn.util import masked_divide

class EsimComparator(torch.nn.Module):

    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 similarity_function: SimilarityFunction = None,
                 dropout: float = 0.5) -> None:
        super().__init__()

        self._encoder = encoder
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._projection_feedforward = projection_feedforward
        self._inference_encoder = inference_encoder
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None
        self._output_feedforward = output_feedforward

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                embedded_premise: torch.Tensor,
                embedded_hypothesis: torch.Tensor,
                premise_mask: torch.Tensor = None,
                hypothesis_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]: # pylint: disable=unused-argument

        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_premise = self.rnn_input_dropout(embedded_premise)
            embedded_hypothesis = self.rnn_input_dropout(embedded_hypothesis)

        # encode premise and hypothesis
        encoded_premise = self._encoder(embedded_premise, premise_mask)
        encoded_hypothesis = self._encoder(embedded_hypothesis, hypothesis_mask)

        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(encoded_premise, encoded_hypothesis)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = masked_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(encoded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(encoded_premise, h2p_attention)

        # the "enhancement" layer
        premise_enhanced = torch.cat([encoded_premise, attended_hypothesis,
                                      encoded_premise - attended_hypothesis,
                                      encoded_premise * attended_hypothesis], dim=-1)
        hypothesis_enhanced = torch.cat([encoded_hypothesis, attended_premise,
                                         encoded_hypothesis - attended_premise,
                                         encoded_hypothesis * attended_premise], dim=-1)

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected_enhanced_premise = self._projection_feedforward(premise_enhanced)
        projected_enhanced_hypothesis = self._projection_feedforward(hypothesis_enhanced)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_premise = self.rnn_input_dropout(projected_enhanced_premise)
            projected_enhanced_hypothesis = self.rnn_input_dropout(projected_enhanced_hypothesis)
        v_ai = self._inference_encoder(projected_enhanced_premise, premise_mask)
        v_bi = self._inference_encoder(projected_enhanced_hypothesis, hypothesis_mask)

        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        v_a_max, _ = replace_masked_values(v_ai, premise_mask.unsqueeze(-1), -1e7).max(dim=1)
        v_b_max, _ = replace_masked_values(v_bi, hypothesis_mask.unsqueeze(-1), -1e7).max(dim=1)

        # #
        # v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(-1), dim=1) / (torch.sum(premise_mask, 1, keepdim=True)+1e-8)
        # v_b_avg = torch.sum(v_bi * hypothesis_mask.unsqueeze(-1), dim=1) / (torch.sum(hypothesis_mask, 1, keepdim=True)+1e-8)

        # #
        v_a_avg = masked_divide(torch.sum(v_ai * premise_mask.unsqueeze(-1), dim=1),
                                torch.sum(premise_mask, 1, keepdim=True))
        v_b_avg = masked_divide(torch.sum(v_bi * hypothesis_mask.unsqueeze(-1), dim=1),
                                (torch.sum(hypothesis_mask, 1, keepdim=True)))

        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v_all = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v_all = self.dropout(v_all)

        # output_hidden = self._output_feedforward(v_all)
        output_hidden = self._output_feedforward(v_all)*premise_mask[:, 0].unsqueeze(-1)*hypothesis_mask[:, 0].unsqueeze(-1)

        return output_hidden

    def get_output_dim(self):
        return self._output_feedforward.get_output_dim()


# Compartments of ESIMComparator modules are below:

# Same for premise and hypothesis
class EsimComparatorLayer1(torch.nn.Module):

    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self._encoder = encoder
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                embedded_sentence: torch.Tensor,
                sentence_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]: # pylint: disable=unused-argument
        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_sentence = self.rnn_input_dropout(embedded_sentence)
        # encode sentence (premise or hypothesis)
        encoded_sentence = self._encoder(embedded_sentence, sentence_mask)
        return encoded_sentence

    def get_output_dim(self):
        return self._encoder.get_output_dim()


class EsimComparatorLayer2(torch.nn.Module):

    def __init__(self,
                 similarity_function: SimilarityFunction = None) -> None:
        super().__init__()
        # Don't use DotProductMatrixAttention() if model wasn't trained exactly with it.
        self._matrix_attention = LegacyMatrixAttention(similarity_function)

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                encoded_premise: torch.Tensor,
                encoded_hypothesis: torch.Tensor) -> Dict[str, torch.Tensor]: # pylint: disable=unused-argument
        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(encoded_premise, encoded_hypothesis)
        return similarity_matrix

    def get_output_dim(self):
        return self._matrix_attention.get_output_dim()


class EsimComparatorLayer3Plus(torch.nn.Module):

    def __init__(self,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 dropout: float = 0.5) -> None:
        super().__init__()
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None
        self._projection_feedforward = projection_feedforward
        self._inference_encoder = inference_encoder
        self._output_feedforward = output_feedforward
        # self._weight_premise_token = weight_premise_token

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                encoded_premise: torch.Tensor,
                encoded_hypothesis: torch.Tensor,
                similarity_matrix: torch.Tensor,
                premise_mask: torch.Tensor = None,
                hypothesis_mask: torch.Tensor = None,
                premise_token_weights: torch.Tensor = None) -> Dict[str, torch.Tensor]: # pylint: disable=unused-argument

        p2h_attention = masked_softmax(similarity_matrix, hypothesis_mask)

        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(encoded_hypothesis, p2h_attention)

        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)

        if premise_token_weights is not None:
            h2p_attention = premise_token_weights.unsqueeze(1) * h2p_attention
            h2p_attention = masked_divide(h2p_attention, h2p_attention.sum(dim=-1, keepdim=True))

        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(encoded_premise, h2p_attention)

        # the "enhancement" layer
        premise_enhanced = torch.cat([encoded_premise, attended_hypothesis,
                                      encoded_premise - attended_hypothesis,
                                      encoded_premise * attended_hypothesis], dim=-1)
        hypothesis_enhanced = torch.cat([encoded_hypothesis, attended_premise,
                                         encoded_hypothesis - attended_premise,
                                         encoded_hypothesis * attended_premise], dim=-1)

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected_enhanced_premise = self._projection_feedforward(premise_enhanced)
        projected_enhanced_hypothesis = self._projection_feedforward(hypothesis_enhanced)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_premise = self.rnn_input_dropout(projected_enhanced_premise)
            projected_enhanced_hypothesis = self.rnn_input_dropout(projected_enhanced_hypothesis)

        v_ai = self._inference_encoder(projected_enhanced_premise, premise_mask)
        v_bi = self._inference_encoder(projected_enhanced_hypothesis, hypothesis_mask)

        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim)
        v_a_max, _ = replace_masked_values(v_ai, premise_mask.unsqueeze(-1), -1e7).max(dim=1)
        v_b_max, _ = replace_masked_values(v_bi, hypothesis_mask.unsqueeze(-1), -1e7).max(dim=1)

        # #
        v_a_avg = masked_divide(torch.sum(v_ai * premise_mask.unsqueeze(-1), dim=1),
                                torch.sum(premise_mask, 1, keepdim=True))
        v_b_avg = masked_divide(torch.sum(v_bi * hypothesis_mask.unsqueeze(-1), dim=1),
                                (torch.sum(hypothesis_mask, 1, keepdim=True)))

        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v_all = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v_all = self.dropout(v_all)

        # output_hidden = self._output_feedforward(v_all)
        output_hidden = self._output_feedforward(v_all)*premise_mask[:, 0].unsqueeze(-1)*hypothesis_mask[:, 0].unsqueeze(-1)

        return output_hidden, h2p_attention

    def get_output_dim(self):
        return self._output_feedforward.get_output_dim()
