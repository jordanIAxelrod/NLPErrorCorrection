import torch
import torch.nn as nn

import TextPreprocess


class CELWithLevenshteinRegularization(nn.Module):
    def __init__(self, words, reg_lambda, num_changes, num_chars):
        """
        Calculates the CE loss with a regularization of levenshtein
        :param dictionary: dictionary of word ids as keys and their possible transformations as keys
        """
        super(CELWithLevenshteinRegularization, self).__init__()

        self.reg_lambda = reg_lambda

        self.num_changes = num_changes

        self.CEL = nn.CrossEntropyLoss()

    def forward(self, pred, gold, levenshtein) -> torch.Tensor:

        levenshtein = (levenshtein - self.num_changes)
        levenshtein[levenshtein < 0] = 0
        reg_terms = levenshtein.reshape(pred.shape[0], pred.shape[1]) / levenshtein.numel() * pred * self.reg_lambda
        regularize = torch.sum(reg_terms)

        return self.CEL(pred, gold) + regularize


