"""
This file trains the error correction models. We will start with the original paper, our baseline model. Then move to
train our transformer model. For the transformer model we will use a custom loss that heavily punishes the predictions
of words that are not possible to get to from the presented corrupt word.

"""
import torch.cuda
import torch.nn as nn
import torch.optim as optim

import tqdm
import os
from CELWithLevenshteinRegularization import CELWithLevenshteinRegularization
import CharTransformer, BiLSTMErrorCorrection, EmbeddingTypes, ErrorCreator
import TextPreprocess

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device='cpu'

def train(model, optimizer, epochs, train_loader, val_loader, loss_function, model_name, num_words):
    prev = 1
    for epoch in range(epochs):
        model.train()
        for i, X in tqdm.tqdm(enumerate(train_loader), total=train_loader.sampler.batch_count()):
            optimizer.zero_grad()
            X, (corrupts, _), levenshtein = X

            pred = model(corrupts).reshape(-1, num_words)

            y = [[TextPreprocess.ids_word(word) for word in sentance.split(' ')] for sentance in X]

            y = torch.LongTensor(y).to(device).reshape(-1)
            if 'transformer' in model_name:
                loss = loss_function(pred, y, levenshtein)
            else:
                loss = loss_function(pred, y)

            loss.backward()

            optimizer.step()

        model.eval()
        with torch.no_grad():
            all_correct = []
            for i, X in enumerate(val_loader):
                X, (corrupts, corrupted), levenshtein = X
                pred = model(corrupts)

                y = [[TextPreprocess.ids_word(word) for word in sentance.split(' ')] for sentance in X]
                y = torch.LongTensor(y).to(device)

                pred = torch.argmax(pred, dim=-1)

                correct = y != pred
                all_correct.append(correct.flatten())

            all_correct = torch.cat(all_correct)

            wer = sum(all_correct) / all_correct.shape[0]
            print(wer)
            if wer < prev:
                try:
                    torch.save(model, f"../models/{model_name}.pt")
                except Exception as e:
                    print(os.getcwd())
                    raise e
                prev = wer

    model = torch.load(f'models/{model_name}.pt')
    return model


def train_model(model_type, is_foreground):
    cwd = os.getcwd()
    print(cwd)
    os.chdir('..')
    if is_foreground:
        TextPreprocess.create_dictionary('stanford')
        words = TextPreprocess.vocab('foreground')
        print(TextPreprocess.textdata['char_type2idx'])
        num_words = len(words)
        name = model_type + '_foreground'
        if 'stanford_train_loader.pt' in os.listdir() and 'stanford_val_loader.pt' in os.listdir():
            train_loader = 'stanford_train_loader.pt'
            val_loader = 'stanford_val_loader.pt'
            loaders_present = True
        else:
            typ = 'stanford'
            loaders_present = False
    else:
        TextPreprocess.create_dictionary('imdb')
        words = TextPreprocess.vocab('background')
        num_words = len(words)
        name = model_type + '_background'
        os.chdir('../data')

        if 'imdb_train_loader' in os.listdir() and 'imdb_train_loader' in os.listdir():
            train_loader = 'imdb_train_loader'
            val_loader = 'imdb_val_loader'
            loaders_present = True
        else:
            typ = 'imdb'
            loaders_present = False
        os.chdir(cwd)
    if model_type.lower() == 'bilstm':
        embedding = EmbeddingTypes.OuterPosBow(' ', 1980, len(TextPreprocess.textdata['char_type2idx'].keys())).to(
            device)
        model = BiLSTMErrorCorrection.BiLSTM(embedding, num_words, 1980, 1500).to(device)
        loss_function = nn.CrossEntropyLoss()

    else:
        num_chars = len(TextPreprocess.textdata['char_type2idx'].keys())
        embedding = EmbeddingTypes.OuterPosBow(' ', 16 * 4, num_chars).to(
            device)
        model = CharTransformer.ErrorCorrector(embedding, num_words, 16 * 4).to(device)
        loss_function = CELWithLevenshteinRegularization(words, 0.01, 1, num_chars).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if loaders_present:
        train_loader = torch.load(train_loader)
        val_loader = torch.load(val_loader)
    else:
        train_loader, _, val_loader = TextPreprocess.create_dataloaders(typ, 10)
    os.chdir(cwd)

    trained_model = train(
        model,
        optimizer,
        30,
        train_loader,
        val_loader,
        loss_function,
        name, num_words
    )


if __name__ == '__main__':
    for model_type in reversed(['bilstm', 'transformer']):
        for is_foreground in (True, False):
            train_model(model_type, is_foreground)
