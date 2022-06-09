"""
    Introduced in the Compact Vision Transformer (CVT) as "SeqPool"
    Escaping the Big Data Paradigm with Compact Transformers - Ali Hassani et al.
    https://arxiv.org/abs/2104.05704
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import HelperModule

class CVTSeqPool(HelperModule):
    def build(self,
            in_dim: int
        ):
        self.in_dim = in_dim
        self.pool = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 3:
            raise ValueError(f"Expected input to have 3 axes. Got {len(x.shape)}")
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Size of embedding dimension ({x.shape[-1]}) did not match expected ({self.in_dim})")

        return (F.softmax(self.pool(x), dim=1).transpose(-1, -2) @ x).squeeze(-2)

if __name__ == '__main__':
    layer = CVTSeqPool(16)
    x = torch.randn(4, 128, 16)
    y = layer(x)

    assert tuple(y.shape) == (4, 16)
