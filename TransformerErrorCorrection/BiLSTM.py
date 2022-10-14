"""
This is the model to replicate the paper and serve as a comparison to my results.
"""
import EmbeddingTypes
import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(self, embedding: EmbeddingTypes.ErrorCorrectionEmbedding, num_words: int, embed_dim):
        super().__init__()

        self.embed = embedding

        self.bilstm = nn.LSTM(
            embed_dim,

        )