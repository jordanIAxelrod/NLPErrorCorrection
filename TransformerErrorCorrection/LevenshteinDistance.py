import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class DamerauLevenshtein(nn.Module):

    def __init__(self, words: torch.Tensor, word_lengths: torch.Tensor, num_chars):
        """
        Initialized the metric
        :param words: A 2-d Tensor of words broken out into characters
            Shape: `(n_words, len_longest_word)
        """
        super(DamerauLevenshtein, self).__init__()
        self.words = words
        print(self.words.shape)
        self.num_words = self.words.shape[0]
        self.word_lengths = word_lengths
        self.max_len = torch.max(word_lengths)
        self.word_lengths_arranged = torch.Tensor([
            [i if i < self.word_lengths[j] else 0 for i in range(self.max_len)]
            for j in range(self.word_lengths.shape[0])
        ]).to(device)
        print(num_chars)
        self.da = F.pad(torch.zeros(num_chars).to(device), (2, 0))

    def _total_combination_eq(self, x, y):
        """
        gets equality between each element in each tensor
        :param x: 2d tensor
        :param y: 1d tensor
        :return: 3d tensor True if equal
        """

        x = x.reshape(x.shape[0], x.shape[1], 1).expand(-1, -1, y.shape[0])
        x = x - y

        return x.bool().long().to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: A 3-d tensor of baches of sequences of words broken out into characters, padded to the largest word
            Shape: `(bsz, seq_len, len_longest_word)
        :return: a tensor of containing the Damerau Levenshtein distance
        between each word in the sequence and every word in the list
            Shape: `(bsz, seq_len, num_words)
        """
        m, sequence_word_lengths = x.min(dim=-1)

        bsz, seq, max_len = x.shape
        assert max_len == self.max_len, "Size of the character paddings don't match"
        d = torch.zeros(bsz, seq, self.num_words, max_len + 2, max_len + 2).to(device)
        max_dist = sequence_word_lengths.unsqueeze(-1).expand(-1, -1, self.num_words) + self.word_lengths
        d[:, :, :, 1, 2:] = self.word_lengths_arranged
        seq_word_len = torch.cat([
            torch.nn.utils.rnn.pad_packed_sequence(
                torch.nn.utils.rnn.pack_sequence(
                    [torch.arange(0, word_len).to(device) for word_len in sequence]
                    , enforce_sorted=False),
                batch_first=True,
                total_length=max_len
            )[0].unsqueeze(0)
            for sequence in sequence_word_lengths
        ], dim=0)

        d[:, :, :, 2:, 1] = seq_word_len.unsqueeze(2)
        d[:, :, :, 0] = max_dist.unsqueeze(-1)
        d[:, :, :, :, 0] = max_dist.unsqueeze(-1)
        for i in range(1, max_len + 1):
            db = torch.zeros(bsz, seq, self.num_words).to(device)
            for j in range(1, max_len + 1):
                k = self.da[self.words[:, j - 2].long()]

                k_idx = k.reshape(1, 1, self.num_words, 1) \
                    .expand(bsz, seq, self.num_words, 1) \
                    .long()
                l = db.clone()
                l_idx = db.reshape(bsz, seq, self.num_words, 1, 1) \
                    .expand(bsz, seq, self.num_words, max_len + 2, 1) \
                    .long()
                d_transpose = torch.gather(d, 4, l_idx).squeeze()
                d_transpose = torch.gather(d_transpose, 3, k_idx).squeeze()
                cost_of_i_j = self._total_combination_eq(x[:, :, i - 2], self.words[:, j - 2])
                db = (1 - cost_of_i_j) * j
                d[:, :, :, i, j] = torch.min(torch.cat([
                    d[:, :, :, i - 1, j].unsqueeze(-1) + 1,
                    d[:, :, :, i, j - 1].unsqueeze(-1) + 1,
                    (d[:, :, :, i - 1, j - 1] + cost_of_i_j).unsqueeze(-1),
                    (d_transpose + (i - k - 1) + 1 + j - l - 1).unsqueeze(-1)
                ], dim=-1
                ), dim=-1)[0]

                self.da[x[:, :, i - 2]] = i
        word_idx = self.word_lengths.reshape(1, 1, self.num_words, 1, 1).expand(bsz, seq, self.num_words, 1, max_len)
        levenshtein = torch.gather(d[:, :, :, 1:, 1:], 3, word_idx)  # (bsz, seq, num_words, 1, max_len)
        sequence_idx = sequence_word_lengths.reshape(bsz, seq, 1, 1).expand(bsz, seq, self.num_words, 1)
        levenshtein = torch.gather(levenshtein.squeeze(), 3, sequence_idx).squeeze()  # (bsz, seq, num_words)
        return levenshtein
