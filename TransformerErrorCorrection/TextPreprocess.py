"""
This file creates and saves a corrupted version of the input.

"""
import torch
from collections import defaultdict, Counter, OrderedDict
import pandas as pd
from sklearn.model_selection import train_test_split
import ErrorCreator
import datasets
import ErrorCreator
import LevenshteinDistance

## Data setup. Please do NOT change any of this.
textdata = {}
prob = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_dictionary(typ):
    if type == 'imdb':
        data = pd.read_csv('IMDB/IMDB Dataset.csv')
    else:
        data = pd.read_csv('stanfordSentimentTreebank/datasetSentences.txt', sep='\t')
    data.rename({'sentence': 'review'}, inplace=True, axis=1)

    data = datasets.Dataset.from_pandas(data)

    # get word types and indices
    min_freq = 2  # any word occuring < min_freq times gets <unk>ed
    word_counter = Counter()
    char_counter = Counter()
    for example in data:
        word_counter.update(example["review"].split(' '))
        for word in example['review'].split(' '):
            char_counter.update(word)

    for val in ErrorCreator.get_qwerty():
        char_counter.update(val)
    word_types = ["<unk>"] + [wtype for (wtype, wcount) in word_counter.most_common()][:10000]
    char_types = ["<unk>"] + [ctype for (ctype, ccount) in char_counter.most_common() if ctype.isalnum()]
    textdata['char_type2idx'] = {chartype: i for i, chartype in enumerate(char_types)}
    textdata['word_type2idx'] = {wordtype: i for i, wordtype in enumerate(word_types)}
    textdata['word_lengths'] = torch.LongTensor([len(word) for word in textdata['word_type2idx'].keys()]).to(device)
    textdata['max_len'] = max(textdata['word_lengths'])
    textdata['word_tensor'] = character_break_down(list(textdata['word_type2idx'].keys())).squeeze().to(device)



def character_break_down(x: list) -> torch.Tensor:
    """
        gets a vectorized embedding of characters in words
        :param x: list of seqences of words
        :return: a 3d tensor containing an embedding of each word
        """
    sequences = torch.LongTensor([
        [
            [ids_char(word[i]) if i < len(word) else len(textdata['char_type2idx']) + 1 for i in range(textdata['max_len'] + 1)]
            for word in seq.split(' ')]
        for seq in x
    ]).to(device)

    return sequences


def ids_word(word):
    return textdata['word_type2idx'][word] if word in textdata['word_type2idx'] else textdata['word_type2idx']["<unk>"]


def ids_char(char):
    return textdata['char_type2idx'][char] if char in textdata['char_type2idx'] else textdata['char_type2idx']["<unk>"]


def collate(batchdictseq):
    batchdict = batchdictseq[0]
    wordseqs = [wordlist  # batchsize x M
                for wordlist in batchdict['review']]
    corrupt = list(zip(*[ErrorCreator.corrupt_sentence(sentence, prob) for sentence in wordseqs]))
    return wordseqs, corrupt


class FeaturizedDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_dicts):
        super().__init__()
        self.list_of_dicts = list_of_dicts

    def __getitem__(self, index):
        if isinstance(index, list):
            batch = {}
            for rulidx in index:
                for key, val in self.list_of_dicts[rulidx].items():
                    if key not in batch:
                        batch[key] = [val]
                    else:
                        batch[key].append(val)
            return batch
        return self.list_of_dicts[index]


class ByLengthSampler(torch.utils.data.Sampler):
    """
    Allows for sampling minibatches of examples all of the same sequence length;
    adapted from https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/13.
    """

    def __init__(self, dataset, key, batchsize, shuffle=True):
        # import ipdb
        # ipdb.set_trace()
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.seqlens = torch.LongTensor([len(example[key].split(' ')) for example in dataset])
        self.nbatches = len(self._generate_batches())

    def _generate_batches(self):
        # shuffle examples
        seqlens = self.seqlens
        perm = torch.randperm(seqlens.size(0)) if self.shuffle else torch.arange(seqlens.size(0))
        batches = []
        len2batch = defaultdict(list)
        for i, seqidx in enumerate(perm):
            seqlen, seqidx = seqlens[seqidx].item(), seqidx.item()
            len2batch[seqlen].append(seqidx)
            if len(len2batch[seqlen]) >= self.batchsize:
                batches.append(len2batch[seqlen][:])
                del len2batch[seqlen]
        # add any remaining batches
        for length, batchlist in len2batch.items():
            if len(batchlist) > 0:
                batches.append(batchlist)
        # shuffle again so we don't always start w/ the most common sizes
        batchperm = torch.randperm(len(batches)) if self.shuffle else torch.arange(len(batches))
        return [batches[idx] for idx in batchperm]

    def batch_count(self):
        return self.nbatches

    def __len__(self):
        return len(self.seqlens)

    def __iter__(self):
        batches = self._generate_batches()
        for batch in batches:
            yield batch


class SingletonSampler(torch.utils.data.Sampler):
    """
    Samples data one by one. To be used for test data.
    """

    def __init__(self, dataset):
        self.nbatches = sum([1 for _ in dataset])

    def batch_count(self):
        return self.nbatches

    def __len__(self):
        return self.nbatches

    def __iter__(self):
        for i in range(self.nbatches):
            yield [i]


def create_dataloaders(type, batchsize):
    if type == 'imdb':
        data = pd.read_csv('IMDB/IMDB Dataset.csv')
    else:
        data = pd.read_csv('stanfordSentimentTreebank/datasetSentences.txt', sep='\t')
    data.rename({'sentence': 'review'}, inplace=True, axis=1)

    train, rest = train_test_split(data, test_size=.3, random_state=0)
    test, val = train_test_split(rest, test_size=.5, random_state=0)
    train = datasets.Dataset.from_pandas(train)
    test = datasets.Dataset.from_pandas(test)
    val = datasets.Dataset.from_pandas(val)
    train_loader = torch.utils.data.DataLoader(train, batch_size=1,
                                               sampler=ByLengthSampler(train, 'review', batchsize, shuffle=True),
                                               collate_fn=collate,
                                               num_workers=3)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1,
                                              sampler=ByLengthSampler(test, 'review', batchsize, shuffle=True),
                                              collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val, batch_size=1,
                                             sampler=ByLengthSampler(val, 'review', batchsize, shuffle=True),
                                             collate_fn=collate)
    return train_loader, val_loader, test_loader


def vocab(param):
    if param == 'foreground':
        data = pd.read_csv('stanfordSentimentTreebank/datasetSentences.txt', sep='\t')
    else:
        data = pd.read_csv('IMDB/IMDB Dataset.csv')

    data.rename({'sentence': 'reviews'}, inplace=True, axis=1)
    print(data)
    data['reviews'] = data.reviews.apply(lambda x: x.split())
    data = datasets.Dataset.from_pandas(data)
    word_counter = Counter()
    char_counter = Counter()
    for example in data:
        word_counter.update(example["reviews"])
        char_counter.update(' '.join(example['reviews']))
    word_types = ["<unk>"] + [wtype for (wtype, wcount) in word_counter.most_common()][:10000]
    char_types = ["<unk>"] + [ctype for (ctype, ccount) in char_counter.most_common()]

    return word_types
