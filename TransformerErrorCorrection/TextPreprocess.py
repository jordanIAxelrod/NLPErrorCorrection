"""
This file creates and saves a corrupted version of the input.

"""
import torch
from collections import defaultdict, Counter, OrderedDict
## Data setup. Please do NOT change any of this.
nerdata = datasets.load_dataset("conll2003")

# get label types and indices
label_types = nerdata["train"].features["ner_tags"].feature.names
label_type2idx = {labeltype: i for i, labeltype in enumerate(label_types)}

# get word types and indices
min_freq = 2 # any word occuring < min_freq times gets <unk>ed
word_counter = Counter()
char_counter = Counter
for example in nerdata["train"]:
    word_counter.update(example["tokens"])
    char_counter.update(' '.join(example['tokens']))
word_types = ["<unk>"] + [wtype for (wtype, wcount) in word_counter.most_common()
                          if wcount >= min_freq]
word_type2idx = {wordtype: i for i, wordtype in enumerate(word_types)}

def ids_word(word):
    return word_type2idx[word] if word in word_type2idx else word_type2idx["<unk>"]

print("labels", label_types, "(total", len(label_types), ")")
print("using a vocabulary of size", len(word_types))

# only keep the tokens and ner tags.
trdata, valdata = nerdata["train"], nerdata["validation"]
trdata.set_format(columns=['tokens', 'ner_tags'])
valdata.set_format(columns=['tokens', 'ner_tags'])

def collate(batchdictseq):
    batchdict = batchdictseq[0]
    wordseqs = torch.LongTensor([[word2idx(word) for word in wordlist] # batchsize x M
                                 for wordlist in batchdict['tokens']])
    tgtseqs = torch.LongTensor(batchdict["ner_tags"]) # these are already indices
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
        self.seqlens = torch.LongTensor([len(example[key]) for example in dataset])
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


def ids_word(word):
    pass

def ids_char(char):
    pass