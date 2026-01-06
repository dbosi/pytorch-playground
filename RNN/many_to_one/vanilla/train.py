import kagglehub
import pandas
import numpy as np
import torch
import torchmetrics
import pathlib


### CONSTANTS ###

CPU_DEVICE = "cpu"
CUDA_DEVICE = "cuda"

MODEL_NAME = pathlib.Path("model.pth")

PADDING_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

TRAIN_PERCENTAGE = 0.6
VAL_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.2

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

EMBEDDING_SIZE = 64
HIDDEN_1_NEURONS = 128

### VARIABLES ###

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)


### DATASET ###

path = pathlib.Path(kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"))
dataframe = pandas.read_csv(path / "IMDB Dataset.csv")

review_col, sentiment_col = tuple(dataframe.columns.array)
dataframe[review_col] = dataframe[review_col].str.replace(r"(<br />|\"|\,|\.|!|\?|\(|\)|;|:|\\|\*|-|/)", " ", regex=True)

token_lists = dataframe[review_col].apply(lambda x: x.lower().split())
tokens = sorted(set(np.concatenate(token_lists.to_numpy())))
tokens.insert(0, PADDING_TOKEN)
tokens.insert(1, UNK_TOKEN)

labels = dataframe[sentiment_col].unique().tolist()

vocab_stoi = {str(word): idx for idx, word in enumerate(tokens)}
vocab_itos = {idx: word for word, idx in vocab_stoi.items()}

label_stoi = {str(word): idx for idx, word in enumerate(labels)}
label_itos = {idx: word for word, idx in label_stoi.items()}

token_idxs = token_lists.apply(lambda x: [vocab_stoi.get(word, vocab_stoi[UNK_TOKEN]) for word in x]).tolist()
max_len = max(len(arr) for arr in token_idxs)
token_idxs_padded = [arr + [vocab_stoi[PADDING_TOKEN]] * (max_len - len(arr)) for arr in token_idxs]

x_tensor = torch.tensor(token_idxs_padded, dtype=torch.long)
y_tensor = torch.tensor(dataframe[sentiment_col].apply(lambda x: label_stoi[x]).tolist(), dtype=torch.long)

n_examples, seq_len = x_tensor.shape

dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)

n_train, n_val, n_test = int(TRAIN_PERCENTAGE * n_examples), int(VAL_PERCENTAGE * n_examples), int(TEST_PERCENTAGE * n_examples)

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)


### MODEL ###


embedding_vector = torch.randn(len(tokens), EMBEDDING_SIZE, dtype=torch.float32, device=device)

wx = torch.randn(EMBEDDING_SIZE, HIDDEN_1_NEURONS, dtype=torch.float32, device=device)
wh = torch.randn(HIDDEN_1_NEURONS, HIDDEN_1_NEURONS, dtype=torch.float32, device=device)
bh = torch.zeros(HIDDEN_1_NEURONS, dtype=torch.float32, device=device)

wo = torch.randn(HIDDEN_1_NEURONS, len(labels), dtype=torch.float32, device=device)
bo = torch.zeros(len(labels), dtype=torch.float32, device=device)

torch.nn.init.xavier_uniform_(embedding_vector)
torch.nn.init.xavier_uniform_(wx)
torch.nn.init.xavier_uniform_(wh)
torch.nn.init.xavier_uniform_(wo)

parameters = [embedding_vector, wx, wh, bh, wo, bo]

for p in parameters:
    p.requires_grad = True


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(labels)).to(device)
mean = torchmetrics.MeanMetric().to(device)


for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}")

    mean.reset()
    accuracy.reset()

    for b, (x, y) in enumerate(train_loader):
        x: torch.Tensor
        y: torch.Tensor
        
        x = x.to(device)
        y = y.to(device)
        
        h = torch.zeros(x.size(0), HIDDEN_1_NEURONS, dtype=torch.float32, device=device)

        embedded_x = embedding_vector[x, :].permute(1, 0, 2)

        for t in range(seq_len):
            zx = embedded_x[t] @ wx
            zh = h @ wh

            h = torch.tanh(zx + zh + bh)
        
        logit_y = h @ wo + bo

        optimizer.zero_grad()

        loss: torch.Tensor = criterion(logit_y, y)
        loss.backward()

        optimizer.step()

        mean.update(loss)
        accuracy.update(logit_y, y)

        print(f"\rtrain_loss: {mean.compute().item():.4f}\t Batch: {b+1}/{len(train_loader)}", end="")

    print(f"\rtrain_loss: {mean.compute().item():.4f}\ttrain_acc: {accuracy.compute().item():.4f}")

    with torch.no_grad():
        mean.reset()
        accuracy.reset()

        for x, y in val_loader:
            x: torch.Tensor
            y: torch.Tensor
            
            x = x.to(device)
            y = y.to(device)
            
            h = torch.zeros(x.size(0), HIDDEN_1_NEURONS, dtype=torch.float32, device=device)

            embedded_x = embedding_vector[x, :].permute(1, 0, 2)

            for t in range(seq_len):
                zx = embedded_x[t] @ wx
                zh = h @ wh

                h = torch.tanh(zx + zh + bh)
            
            logit_y = h @ wo + bo

            loss: torch.Tensor = criterion(logit_y, y)
            
            mean.update(loss)
            accuracy.update(logit_y, y)
        
        print(f"val_loss: {mean.compute().item():.4f}\tval_acc: {accuracy.compute().item():.4f}\n")

state_dict = {
    "embedding_vector": embedding_vector,
    "wx": wx,
    "wh": wh,
    "bh": bh,
    "wo": wo,
    "bo": bo
}


torch.save(state_dict, MODEL_NAME)

### TESTING MODEL ###

with torch.no_grad():
        mean.reset()
        accuracy.reset()

        for x, y in test_loader:
            x: torch.Tensor
            y: torch.Tensor
            
            x = x.to(device)
            y = y.to(device)
            
            h = torch.zeros(x.size(0), HIDDEN_1_NEURONS, dtype=torch.float32, device=device)

            embedded_x = embedding_vector[x, :].permute(1, 0, 2)

            for t in range(seq_len):
                zx = embedded_x[t] @ wx
                zh = h @ wh

                h = torch.tanh(zx + zh + bh)
            
            logit_y = h @ wo + bo

            loss: torch.Tensor = criterion(logit_y, y)
            
            mean.update(loss)
            accuracy.update(logit_y, y)
        
        print(f"\ntest_loss: {mean.compute().item():.4f}\ttest_acc: {accuracy.compute().item():.4f}\n")