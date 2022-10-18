"""
This is the model to replicate the paper and serve as a comparison to my results.
"""
import EmbeddingTypes
import torch.nn as nn
import torch


class BiLSTM(nn.Module):

    def __init__(self, embedding: EmbeddingTypes.ErrorCorrectionEmbedding, num_words: int, embed_dim, hidden_dim):
        super().__init__()

        self.embed = embedding

        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, num_words // 2),
            nn.ReLU(),
            nn.Linear(num_words // 2, num_words),
            nn.Softmax(dim=2)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.embed(X)
        X = self.bilstm(X)

        return self.head(X)
