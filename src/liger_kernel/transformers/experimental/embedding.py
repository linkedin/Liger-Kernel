from typing import Optional

import torch
import torch.nn as nn

from liger_kernel.ops.experimental.embedding import LigerEmbeddingFunction


class LigerEmbedding(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, indices):
        embedded = LigerEmbeddingFunction.apply(self.weight, indices)
        if self.padding_idx is not None:
            embedded = embedded.clone()
            embedded[indices == self.padding_idx] = 0
        return embedded
