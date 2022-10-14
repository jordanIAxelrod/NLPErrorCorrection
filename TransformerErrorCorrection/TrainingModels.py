
"""
This file trains the error correction models. We will start with the original paper, our baseline model. Then move to
train our transformer model. For the transformer model we will use a custom loss that heavily punishes the predictions
of words that are not possible to get to from the presented corrupt word.

"""
import torch.cuda
import torch.nn as nn
import CharTransformer, BiLSTMErrorCorrection, CELWithPossibilityRegularization, EmbeddingTypes
import TextPreprocess

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train(model, optimizer, epochs, train_loader, val_loader, loss_function, model_name):

    prev = 1
    for epoch in range(epochs):
        model.train()
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(X)

            y = [[ids_word(word) for word in sentance] for sentance in y]

            y = torch.LongTensor(y).to(device)

            loss = loss_function(pred, y)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            all_correct = []
            for i, (X, y) in enumerate(val_loader):

                pred = model(X)

                y = [[ids_word(word) for word in sentance] for sentance in y]
                y = torch.LongTensor(y).to(device)

                pred = torch.argmax(pred, dim=-1)

                correct = y != pred
                all_correct.append(correct.flatten())

            all_correct = torch.cat(all_correct)

            wer = sum(all_correct) / all_correct.shape[0]
            if wer < prev:
                torch.save(model.state_dict(), f"models/{model_name}.pt")
                prev = wer

    model = torch.load(f'models/{model_name}.pt')
    return model


def train_background(model_type):
    num_words = TextPreprocess.background_num_words
    embedding = EmbeddingTypes.OuterPosBow(' -_')
    if model_type.lower() == 'bilstm':
        model = BiLSTMErrorCorrection.BiLSTM(embedding, num_words, 198, 50)
    else:
        model = CharTransformer.ErrorCorrector(embedding, num_words, )

