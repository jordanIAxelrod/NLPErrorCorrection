import torch
import torch.nn as nn

import TextPreprocess
import LevenshteinDistance


class CELWithLevenshteinRegularization(nn.Module):
    def __init__(self, words, reg_lambda, num_changes, num_chars):
        """
        Calculates the CE loss with a regularization of levenshtein
        :param dictionary: dictionary of word ids as keys and their possible transformations as keys
        """
        super(CELWithLevenshteinRegularization, self).__init__()
        self.word_lens = [len(word) for word in words]
        self.max_len = max(self.word_lens)
        self.words = self.character_break_down(words).squeeze() # (num_words, max_len)

        self.reg_lambda = reg_lambda

        self.num_changes = num_changes
        self.damerau_levenshtein = LevenshteinDistance.DamerauLevenshtein(
            self.words,
            torch.LongTensor(self.word_lens),
            num_chars
        )

        self.CEL = nn.CrossEntropyLoss()

    def forward(self, pred, gold, X) -> torch.Tensor:
        X = self.character_break_down(X)
        levenshtein = self.damerau_levenshtein(X)
        levenshtein = (torch.Tensor(levenshtein) - self.num_changes).to(pred.device)
        levenshtein[levenshtein < 0] = 0
        reg_terms = levenshtein.reshape(pred.shape[0], pred.shape[1]) / levenshtein.numel() * pred * self.reg_lambda
        regularize = torch.sum(reg_terms)

        return self.CEL(pred, gold) + regularize

    def character_break_down(self, x: list) -> torch.Tensor:
        """
        gets a vectorized embedding of characters in words
        :param x: list of seqences of words
        :return: a 3d tensor containing an embedding of each word
        """
        sequences = torch.LongTensor([
            [
                [TextPreprocess.ids_char(seq) + 1 if i < len(word) else 0 for i in range(self.max_len)]
                for word in seq.split(' ')]
            for seq in x
        ])

        return sequences
