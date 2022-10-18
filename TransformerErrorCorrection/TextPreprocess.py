"""
This file creates and saves a corrupted version of the input.

"""
import torch
from collections import defaultdict, Counter, OrderedDict
import pandas as pd
from sklearn.model_selection import train_test_split
import ErrorCreator
import datasets

## Data setup. Please do NOT change any of this.
word_type2idx = None
char_type2idx = None

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
        word_counter.update(example["review"].split())
        char_counter.update(example['review'])
    word_types = ["<unk>"] + [wtype for (wtype, wcount) in word_counter.most_common()
                              if wcount >= min_freq]
    char_types = ["<unk>"] + [ctype for (ctype, ccount) in char_counter.most_common()]
    char_type2idx = {chartype: i for i, chartype in enumerate(char_types)}
    word_type2idx = {wordtype: i for i, wordtype in enumerate(word_types)}


def ids_word(word):
    return word_type2idx[word] if word in word_type2idx else word_type2idx["<unk>"]


def ids_char(char):
    return char_type2idx[char] if char in char_type2idx else char_type2idx["<unk>"]


def collate(batchdictseq):
    batchdict = batchdictseq[0]
    wordseqs = torch.LongTensor([[ids_word(word) for word in wordlist]  # batchsize x M
                                 for wordlist in batchdict['tokens']])
    tgtseqs = torch.LongTensor(batchdict["ner_tags"])  # these are already indices
    return wordseqs, tgtseqs


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
    data.rename({'sentence': 'review'}, inplace=True)

    train, rest = train_test_split(data, test_size=.3, random_state=0)
    test, val = train_test_split(rest, test_size=.5, random_state=0)
    train = datasets.Dataset.from_pandas(train)['train']
    test = datasets.Dataset.from_pandas(test)['train']
    val = datasets.Dataset.from_pandas(val)['train']
    train_loader = torch.utils.data.DataLoader(train, batch_size=1,
                                               sampler=ByLengthSampler(train, 'tokens', batchsize, shuffle=True),
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1,
                                              sampler=ByLengthSampler(test, 'tokens', batchsize, shuffle=True),
                                              collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val, batch_size=1,
                                             sampler=ByLengthSampler(val, 'tokens', batchsize, shuffle=True),
                                             collate_fn=collate)
    return train_loader, val_loader, test_loader


def vocab(param):
    if param == 'foreground':
        data = pd.read_csv('stanfordSentimentTreebank/datasetSentences.txt', sep='\t')
    else:
        data = pd.read_csv('IMDB/IMDB Dataset.csv')

    data.rename({'sentence': 'reviews'}, inplace=True)

    data['review'] = data.review.apply(lambda x: x.split())
    word_counter = Counter()
    char_counter = Counter()
    for example in data:
        word_counter.update(example["reviews"])
        char_counter.update(' '.join(example['reviews']))
    word_types = ["<unk>"] + [wtype for (wtype, wcount) in word_counter.most_common()]
    char_types = ["<unk>"] + [ctype for (ctype, ccount) in char_counter.most_common()]

    return word_types
