import os
import torch
from torch import nn


class EmbeddingModel(nn.Module):
    def __init__(self, num_players):
        super().__init__()
        self.embedding = nn.Embedding(num_players, 100)

    def forward(self, x):
        return self.embedding(x)
    