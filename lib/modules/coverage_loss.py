from overrides import overrides
import torch

from allennlp.nn.util import masked_mean, replace_masked_values
from allennlp.common.registrable import Registrable

class CoverageLoss(torch.nn.Module, Registrable):

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                premises_relevance_logits: torch.Tensor,
                premises_presence_mask: torch.Tensor,
                relevance_presence_mask: torch.Tensor) -> torch.Tensor: # pylint: disable=unused-argument
        raise NotImplementedError


@CoverageLoss.register("bce")
class BceCoverageLoss(CoverageLoss):

    def __init__(self):
        super().__init__()
        self._loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                premises_relevance_logits: torch.Tensor,
                premises_presence_mask: torch.Tensor,
                relevance_presence_mask: torch.Tensor) -> torch.Tensor: # pylint: disable=unused-argument
        premises_relevance_logits = replace_masked_values(premises_relevance_logits, premises_presence_mask, -1e10)
        binary_losses = self._loss(premises_relevance_logits, relevance_presence_mask)
        coverage_losses = masked_mean(binary_losses, premises_presence_mask, dim=1)
        coverage_loss = coverage_losses.mean()
        return coverage_loss
