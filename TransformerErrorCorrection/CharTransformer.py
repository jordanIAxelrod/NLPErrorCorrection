import torch
import torch.nn as nn
import EmbeddingTypes


class ErrorCorrector(nn.Module):
    """
    This model takes sentences with errors in all words >4 characters long.
    It tries to map these incorrect words to the original. The embedding type
    is variable. It can either be character based, first last and bag, word or charachter word.
    """

    def __init__(self, embedding: EmbeddingTypes.ErrorCorrectionEmbedding, num_words: int, embed_dim):
        super().__init__()
        self.embed = embedding
        self.transformer = nn.TransformerEncoderLayer(embed_dim, embed_dim // 16, 512, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_words // 2),
            nn.ReLU(),
            nn.Linear(num_words // 2, num_words),
            nn.Softmax(dim=2)
        )

    def forward(self, sntcs: list) -> torch.Tensor:
        X = self.embed(sntcs)
        X = self.transformer(X)
        if len(X.shape) < 3:
            print(X)
        return self.head(X)
