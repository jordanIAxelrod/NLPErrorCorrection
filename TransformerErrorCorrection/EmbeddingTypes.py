"""
This file defines four embedding types, outer positions + bow, character, word, and word char
"""
import numpy as np
import torch
import torch.nn as nn
import re
import torch.nn.functional as F
from TextPreprocess import ids_char, ids_word

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ErrorCorrectionEmbedding(nn.Module):
    def __init__(self, stop_chars, embed_dim, input_dim):
        super(ErrorCorrectionEmbedding, self).__init__()
        self.stop_chars = '|'.join(stop_chars)
        self.embed = nn.Linear(input_dim, embed_dim, bias=False)

    def forward2(self, sntcs: list):
        sntcs = [''.join(letter for letter in sntc if letter.isalpha()) for sntc in sntcs]
        word_lists = [re.split(self.stop_chars, sntc) for sntc in sntcs]
        return word_lists


class OuterPosBow(ErrorCorrectionEmbedding):
    def __init__(self, stop_chars: str, embed_dim: int, num_chars):
        super().__init__(
            stop_chars,
            embed_dim // 3,
            num_chars
        )
        self.num_chars = num_chars
        self.dim_add = embed_dim % 3
        self.relu = nn.ReLU()

    def forward(self, sntcs: torch.Tensor) -> torch.Tensor:

        word_lengths = self.relu(torch.argmax(sntcs, dim=2) - 1)
        ends = torch.gather(sntcs, 2, word_lengths.unsqueeze(2)).squeeze(2)

        sntcs = torch.scatter(sntcs, 2, word_lengths.unsqueeze(2), 0).long()
        sntcs[:, :, -1] = ends

        word_list = F.one_hot(sntcs)[:, :, :, :self.num_chars].float()
        word_list = self.embed(word_list)
        bow = torch.sum(word_list[:, :, 1:-1], dim=2).squeeze(2)
        print(bow.shape)
        X = torch.cat([word_list[:, :, 0], bow, word_list[:, :, -1]], dim=2).squeeze(2)
        X = F.pad(X, (self.dim_add, 0))
        return X


class CharacterEmbedding(ErrorCorrectionEmbedding):
    def __init__(self, stop_chars: str, embed_dim, isTransformer: bool, num_chars):
        super().__init__(
            stop_chars,
            embed_dim,
            num_chars
        )
        self.isTransformer = isTransformer
        self.num_chars = num_chars
        if isTransformer:
            embed_layer = nn.TransformerEncoderLayer(
                embed_dim,
                embed_dim // 16,
                dim_feedforward=512,
                batch_first=True
            )
            self.embed_dim = embed_dim
            self.embed = nn.TransformerEncoder(embed_layer, 1)

        else:
            self.embed = nn.LSTM(
                num_chars,
                num_chars,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )
        self.head = nn.Sequential(
            nn.Linear(num_chars, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def _make_mask(self, x):
        mask = self.embed_dim - torch.sum(x == torch.zeros(self.embed_dim), dim=-1)

        mask[mask > 0] = -np.Inf

        return mask

    def forward(self, sntcs: list):

        word_list = super().forward2(sntcs)

        char_list = torch.LongTensor([
            [
                [ids_char(word[char])
                 if char < len(word)
                 else self.num_chars + 1
                 for char in word]
                for word in l]
            for l in word_list
        ]).to(device)

        char_list = F.one_hot(char_list, num_classes=self.num_chars)

        embeddings = []
        for i in char_list.shape[1]:
            X = char_list[:, i]
            if self.isTransformer:
                X = self.linear(X)
                mask = self._make_mask(X)
                X = self.embed(X, mask)

            else:
                X = self.embed(X)
            X = self.head(X)
            X = torch.sum(X, dim=1)
            embeddings.append(X)

        X = torch.cat(embeddings, dim=1)
        return X


class WordEmbedding(ErrorCorrectionEmbedding):

    def __init__(self, stop_chars, embed_dim, num_words):
        super().__init__(
            stop_chars,
            embed_dim,
            num_words
        )

    def forward(self, word_list):
        word_list = super().forward2(word_list)

        word_list = torch.LongTensor([[ids_word(word) for word in sentence] for sentence in word_list])

        X = F.one_hot(word_list)

        return self.embed(X)


class WordCharEmbed(ErrorCorrectionEmbedding):
    def __init__(self, stop_chars, embed_dim, num_words, isTransformer):
        self.word_embed = WordEmbedding(
            stop_chars,
            embed_dim,
            num_words
        )

        self.char_embed = CharacterEmbedding(
            stop_chars,
            embed_dim,
            isTransformer
        )

    def forward(self, sntcs: list) -> torch.Tensor:
        return self.word_embed(sntcs) + self.char_embed(sntcs)
