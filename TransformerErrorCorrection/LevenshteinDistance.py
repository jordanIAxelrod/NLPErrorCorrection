import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class DamerauLevenshtein(nn.Module):

    def __init__(self, words: torch.Tensor, word_lengths: torch.Tensor, num_chars):
        """
        Initialized the metric
        :param words: A 2-d Tensor of words broken out into characters
            Shape: `(n_words, len_longest_word)
        """
        super(DamerauLevenshtein, self).__init__()
        self.words = words
        self.num_words = self.words.shape[0]
        self.word_lengths = word_lengths
        print(word_lengths.shape)
        self.max_len = torch.max(word_lengths)
        self.word_lengths_arranged = torch.Tensor([
            [i if i < self.word_lengths[j] + 1 else 0 for i in range(self.max_len + 1)]
            for j in range(self.word_lengths.shape[0])
        ]).to(device)
        self.num_chars = num_chars

    def _total_combination_eq(self, x, y):
        """
        gets equality between each element in each tensor
        :param x: 2d tensor
        :param y: 1d tensor
        :return: 3d tensor True if equal
        """
        x = x.reshape(x.shape[0], x.shape[1], 1).expand(-1, -1, y.shape[0])
        z = x - y

        return z.bool().long().to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: A 3-d tensor of baches of sequences of words broken out into characters, padded to the largest word
            Shape: `(bsz, seq_len, len_longest_word)
        :return: a tensor of containing the Damerau Levenshtein distance
        between each word in the sequence and every word in the list
            Shape: `(bsz, seq_len, num_words)
        """
        m, sequence_word_lengths = x.max(dim=-1)
        bsz, seq, max_len = x.shape
        da = torch.zeros(bsz, seq, self.num_chars + 2).to(device)

        assert max_len - 1 == self.max_len, "Size of the character paddings don't match"
        d = torch.zeros(bsz, seq, self.num_words, self.max_len + 2, self.max_len + 2).to(device)
        max_dist = sequence_word_lengths.unsqueeze(-1).expand(-1, -1, self.num_words) + self.word_lengths
        d[:, :, :, 1, 1:] = self.word_lengths_arranged

        seq_word_len = torch.cat([
            torch.nn.utils.rnn.pad_packed_sequence(
                torch.nn.utils.rnn.pack_sequence(
                    [torch.arange(0, word_len + 1).to(device) for word_len in sequence]
                    , enforce_sorted=False),
                batch_first=True,
                total_length=max_len
            )[0].unsqueeze(0)
            for sequence in sequence_word_lengths
        ], dim=0)
        d[:, :, :, 1:, 1] = seq_word_len.unsqueeze(2)
        d[:, :, :, 0] = max_dist.unsqueeze(-1)
        d[:, :, :, :, 0] = max_dist.unsqueeze(-1)
        for i in range(1, max_len):
            db = torch.zeros(bsz, seq, self.num_words).to(device).long()
            for j in range(1, max_len):
                idx = self.words[:, j - 1].reshape(1, 1, -1, 1).expand(bsz, seq, self.num_words, 1)
                k = torch.gather(
                    da.reshape(bsz, seq, 1, self.num_chars + 2).expand(bsz, seq, self.num_words, self.num_chars + 2),
                    3,
                    idx
                )
                k_idx = k.long()
                l = db.clone()
                l_idx = db.reshape(bsz, seq, self.num_words, 1, 1) \
                    .expand(bsz, seq, self.num_words, self.max_len + 2, 1) \
                    .long()
                d_transpose = torch.gather(d, 4, l_idx).squeeze(4)
                d_transpose = torch.gather(d_transpose, 3, k_idx).squeeze(3)
                cost_of_i_j = self._total_combination_eq(x[:, :, i - 1], self.words[:, j - 1])
                db[cost_of_i_j == 0] = j

                d[:, :, :, i + 1, j + 1] = torch.min(torch.cat([
                    d[:, :, :, i, j + 1].unsqueeze(-1) + 1,
                    d[:, :, :, i + 1, j].unsqueeze(-1) + 1,
                    (d[:, :, :, i, j] + cost_of_i_j).unsqueeze(-1),
                    (d_transpose + (i - k.squeeze(-1) - 1) + 1 + j - l - 1).unsqueeze(-1)
                ], dim=-1
                ), dim=-1)[0]
            da.scatter_(2, x[:, :, i - 1].unsqueeze(2), i)
        word_idx = self.word_lengths \
                       .reshape(1, 1, self.num_words, 1, 1) \
                       .expand(bsz, seq, self.num_words, max_len + 1, 1) + 1
        levenshtein = torch.gather(d, 4, word_idx)  # (bsz, seq, num_words, 1, max_len)
        sequence_idx = sequence_word_lengths.reshape(bsz, seq, 1, 1).expand(bsz, seq, self.num_words, 1) + 1
        levenshtein = torch.gather(levenshtein.squeeze(4), 3, sequence_idx).squeeze(3)  # (bsz, seq, num_words)
        return levenshtein
