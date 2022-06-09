"""
    Just simple mean pooling.
"""
import torch

from .utils import HelperModule

class MeanSeqPool(HelperModule):
    def build(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)
