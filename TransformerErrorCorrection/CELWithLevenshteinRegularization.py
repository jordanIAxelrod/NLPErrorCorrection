import torch
import torch.nn as nn

from Levenshtein import distance



class CELWithLevenshteinRegularization(nn.Module):
    def __init__(self, words, reg_lambda, num_changes):
        """

        :param dictionary: dictionary of word ids as keys and their possible transformations as keys
        """
        self.words = words

        self.reg_lambda = reg_lambda

        self.num_changes = num_changes

        self.CEL = nn.CrossEntropyLoss()

    def forward(self, pred, gold, X):
        levenshtein = []
        for batch in range(X.size(0)):
            b = []
            for word1 in range(X.size(1)):
                w1 = []
                for word2 in self.words:
                    w1.append(distance(X[batch, word1], word2))
                b.append(w1)
            levenshtein.append(b)

        levenshtein = (torch.Tensor(levenshtein) - self.num_changes).to(pred.device)
        levenshtein[levenshtein < 0] = 0
        regularize = torch.sum(levenshtein * pred * self.reg_lambda)

        return self.CEL(pred, gold) + regularize
