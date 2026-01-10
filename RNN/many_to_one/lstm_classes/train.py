import torch
import torchtext
import torchmetrics
import pandas
import numpy as np
import kagglehub
import pathlib
import typing
import collections
import sklearn.model_selection
import utils.dataset
import utils.model

### CONSTANTS ###

MODEL_NAME = pathlib.Path("model.pth")

DATASET_HANDLE = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
DATASET_NAME = "IMDB Dataset.csv"

REGEX_STRING = r"(<br />|\"|\,|\.|!|\?|\(|\)|;|:|\\|\*|-|/)"

CPU_DEVICE = "cpu"
CUDA_DEVICE = "cuda"

PADDING_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"

VAL_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.2

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

BINARY_THRESHOLD = 0.5

### VARIABLES ###

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)
tokens_counter = collections.Counter()


### FUNCTIONS ###

def yield_tokens(text_list: list[str]) -> typing.Generator[list[str], None, None]:
    for text in text_list:
        yield text.lower().split()

def text_pipeline(text: str, vocab: torchtext.vocab.Vocab) -> list[int]:
    return [vocab[token] for token in text.lower().split()]


### DATASET ###

path = pathlib.Path(kagglehub.dataset_download(DATASET_HANDLE))

dataset_dataframe = pandas.read_csv(path / DATASET_NAME)
review_col, sentiment_col = tuple(dataset_dataframe.columns.to_list())

reviews, sentiments = dataset_dataframe[review_col], dataset_dataframe[sentiment_col]

reviews_clean = reviews.str.replace(REGEX_STRING, " ", regex=True).tolist()

train_val_x, test_x, train_val_y, test_y = sklearn.model_selection.train_test_split(reviews_clean, sentiments.tolist(), test_size=TEST_PERCENTAGE)
train_x, val_x, train_y, val_y = sklearn.model_selection.train_test_split(train_val_x, train_val_y, test_size=VAL_PERCENTAGE)

for tokens_list in yield_tokens(train_x):
    tokens_counter.update(tokens_list)


reviews_vocab = torchtext.vocab.Vocab(tokens_counter, specials=[PADDING_TOKEN, UNKNOWN_TOKEN], min_freq=2)

labels = sorted(sentiments.unique())
labels_stoi = {str(sentiment): idx for idx, sentiment in enumerate(labels)}
labels_itos = {idx: sentiment for sentiment, idx in labels_stoi.items()}


train_reviews_idx = [torch.tensor(text_pipeline(review, reviews_vocab), dtype=torch.long) for review in train_x]
train_reviews_y = torch.tensor([labels_stoi[label] for label in train_y], dtype=torch.float)

val_reviews_idx = [torch.tensor(text_pipeline(review, reviews_vocab), dtype=torch.long) for review in val_x]
val_reviews_y = torch.tensor([labels_stoi[label] for label in val_y], dtype=torch.float)

test_reviews_idx = [torch.tensor(text_pipeline(review, reviews_vocab), dtype=torch.long) for review in test_x]
test_reviews_y = torch.tensor([labels_stoi[label] for label in test_y], dtype=torch.float)

train_dataset = utils.dataset.TextDataset(train_reviews_idx, train_reviews_y)
val_dataset = utils.dataset.TextDataset(val_reviews_idx, val_reviews_y)
test_dataset = utils.dataset.TextDataset(test_reviews_idx, test_reviews_y)


def collate_func(batch: list[tuple[list[torch.Tensor], torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    texts, labels = zip(*batch)

    texts: tuple[list[torch.Tensor], ...] 
    labels: tuple[torch.Tensor, ...] 

    lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)

    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, padding_value=reviews_vocab[PADDING_TOKEN], batch_first=True)

    labels = torch.tensor(labels, dtype=torch.float)

    return padded_texts, lengths, labels


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_func)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_func)


### MODEL ###

def forward_pass(loader_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor], backward: bool, optimizer: typing.Optional[torch.optim.Optimizer] = None) -> tuple[torch.Tensor, torch.Tensor]:
    x, l, y = loader_data

    o: torch.Tensor = model(x, l)
    loss: torch.Tensor = criterion(o, y)

    if backward and optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss, o


model = utils.model.ModelLSTM(len(reviews_vocab), reviews_vocab[PADDING_TOKEN], n_classes=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()
accuracy = torchmetrics.Accuracy(task="binary").to(device)
mean = torchmetrics.MeanMetric().to(device)

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch+1}")

    accuracy.reset()
    mean.reset()

    model.train()

    for b, (x, l, y) in enumerate(train_loader):
        x: torch.Tensor
        l: torch.Tensor
        y: torch.Tensor

        x = x.to(device)
        l = l.to(device)
        y = y.to(device)

        loss, o = forward_pass((x, l, y), True, optimizer)
        
        mean.update(loss)

        predictions = (torch.sigmoid(o) > BINARY_THRESHOLD).float()
        accuracy.update(predictions, y)

        print(f"\rtrain_loss: {mean.compute().item():.4f}\t Batch: {b+1}/{len(train_loader)}", end="")

    print(f"\rtrain_loss: {mean.compute().item():.4f}\ttrain_acc: {accuracy.compute().item():.4f}")

    with torch.no_grad():
        accuracy.reset()
        mean.reset()

        model.eval()

        for b, (x, l, y) in enumerate(val_loader):
            x: torch.Tensor
            l: torch.Tensor
            y: torch.Tensor

            x = x.to(device)
            l = l.to(device)
            y = y.to(device)

            loss, o = forward_pass((x, l, y), False)
            
            mean.update(loss)
            
            predictions = (torch.sigmoid(o) > BINARY_THRESHOLD).float()
            accuracy.update(predictions, y)

        print(f"val_loss: {mean.compute().item():.4f}\tval_acc: {accuracy.compute().item():.4f}\n")


state_dict = model.state_dict()
torch.save(state_dict, MODEL_NAME)


with torch.no_grad():
    accuracy.reset()
    mean.reset()

    model.eval()

    for b, (x, l, y) in enumerate(test_loader):
        x: torch.Tensor
        l: torch.Tensor
        y: torch.Tensor

        x = x.to(device)
        l = l.to(device)
        y = y.to(device)

        loss, o = forward_pass((x, l, y), False)
        
        mean.update(loss)
        
        predictions = (torch.sigmoid(o) > BINARY_THRESHOLD).float()
        accuracy.update(predictions, y)

    print(f"\ntest_loss: {mean.compute().item():.4f}\ttest_acc: {accuracy.compute().item():.4f}")


###
#   test_loss: 0.3447       test_acc: 0.8610 
###