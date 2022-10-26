import fastDamerauLevenshtein
import TransformerErrorCorrection.LevenshteinDistance as ld

import torch

cuda = 'cpu'


def test_levenshtien(word1, word2):
    true_distance = fastDamerauLevenshtein.damerauLevenshtein(word1, word2, similarity=False)
    word2tensor = char_tensor(word2, max([len(word1), len(word2)]) - len(word2) + 1).unsqueeze(0).to(cuda)
    word1tensor = char_tensor(word1, max([len(word1), len(word2)]) - len(word1) + 1).unsqueeze(0).to(cuda)
    words = torch.cat([word2tensor, word1tensor], dim=0)
    print('w',words.shape)
    ld_distance = ld.DamerauLevenshtein(words, torch.LongTensor([len(word2), len(word1)]).to(cuda), 26).to(cuda)
    pred_distance = ld_distance(word1tensor.unsqueeze(0))
    print('pred',pred_distance)
    print('true', true_distance)
    assert pred_distance[0,0,0] == true_distance


def char_tensor(word, pad):
    char_dict = {char: i for i, char in enumerate('qwertyuiopasdfghjklzxcvbnm')}

    x = [char_dict[char] for char in word]

    if pad:
        padding = [len(char_dict) + 1 for _ in range(pad)]
        x.extend(padding)

    return torch.LongTensor(x)


def test_levenshtien_list(word_list):
    true_distance = fastDamerauLevenshtein.damerauLevenshtein(word_list[0], word_list[-1], similarity=False)
    max_len = len(max(word_list, key = lambda x: len(x))) + 1
    tensor = [char_tensor(word, max_len - len(word)).unsqueeze(0) for word in word_list]
    tensor = torch.cat(tensor, dim=0).to('%s' % cuda)
    ld_distance = ld.DamerauLevenshtein(tensor, torch.LongTensor([len(word) for word in word_list]).to(cuda), 26).to(
        cuda)
    pred_distance = ld_distance(tensor.unsqueeze(0))

    assert pred_distance[0, 0, -1] == true_distance




if __name__ =='__main__':
    test_levenshtien_list(['hien', 'bye', 'goodluck', 'goodness', 'higmhness', 'ihness'])
    test_levenshtien('hello', 'hleol')


    # random

    import string, random
    ranges = [random.randint(10, 15) for _ in range(5)]
    list_o_words = [

            ''.join([random.choice(string.ascii_lowercase) for _ in range(num)])
            for num in ranges
    ]
    test_levenshtien_list(list_o_words)


