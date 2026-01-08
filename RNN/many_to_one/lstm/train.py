import numpy as np
import torch
import torchmetrics
import pandas
import pathlib
import kagglehub


### CONSTANTS ###

MODEL_NAME = pathlib.Path("model.pth")

CPU_DEVICE = "cpu"
CUDA_DEVICE = "cuda"

DATASET_HANDLE = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
DATASET_NAME = "IMDB Dataset.csv"

PADDING_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"

TRAIN_PERCENTAGE = 0.6
VAL_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.2

BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.001

EMBEDDING_SIZE = 64
LAYER_NEURONS = 64


### VARIABLES ###

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)


### DATASET ###

path = pathlib.Path(kagglehub.dataset_download(DATASET_HANDLE))
dataframe = pandas.read_csv(path / DATASET_NAME)

review_col, sentiment_col = tuple(dataframe.columns.tolist())

reviews = dataframe[review_col]
sentiments = dataframe[sentiment_col]

reviews_clean = reviews.str.replace(r"(<br />|\"|\,|\.|!|\?|\(|\)|;|:|\\|\*|-|/)", " ", regex=True)


reviews_tokens_lists = reviews_clean.apply(lambda x: x.lower().split())
reviews_tokens = sorted(set(np.concatenate(reviews_tokens_lists.to_numpy())))
reviews_tokens.insert(0, PADDING_TOKEN)
reviews_tokens.insert(1, UNKNOWN_TOKEN)

review_tokens_stoi = {str(word): idx for idx, word in enumerate(reviews_tokens)}
review_tokens_itos = {idx: word for word, idx in review_tokens_stoi.items()}


sentiment_labels = sorted(dataframe[sentiment_col].unique().tolist())
sentiment_labels_stoi = {str(word): idx for idx, word in enumerate(sentiment_labels)}
sentiment_labels_itos = {idx: word for word, idx in sentiment_labels_stoi.items()}


reviews_idx_tokens = reviews_tokens_lists.apply(lambda words_list: [review_tokens_stoi.get(word, review_tokens_stoi[UNKNOWN_TOKEN]) for word in words_list]).tolist()
max_len = max([len(tokens_list) for tokens_list in reviews_idx_tokens])
padded_reviews_idx_tokens = [tokens_list + [review_tokens_stoi[PADDING_TOKEN]] * (max_len - len(tokens_list)) for tokens_list in reviews_idx_tokens]


sentiments_idx = sentiments.apply(lambda sentiment: sentiment_labels_stoi[sentiment]).tolist()


x_tensor = torch.tensor(padded_reviews_idx_tokens, dtype=torch.long)
y_tensor = torch.tensor(sentiments_idx, dtype=torch.long)

n_examples, seq_len = x_tensor.shape
n_train = int(n_examples * TRAIN_PERCENTAGE)
n_val = int(n_examples * VAL_PERCENTAGE)
n_test = n_examples - n_train - n_val

dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)


### MODEL ###

embedding_vector = torch.empty(
    len(reviews_tokens),
    EMBEDDING_SIZE,
    device=device
)

torch.nn.init.normal_(embedding_vector, mean=0.0, std=0.01)

embedding_vector[review_tokens_stoi[PADDING_TOKEN]].zero_()

f_x_w = torch.randn(EMBEDDING_SIZE, LAYER_NEURONS, dtype=torch.float32, device=device)
f_h_w = torch.randn(LAYER_NEURONS, LAYER_NEURONS, dtype=torch.float32, device=device)
f_b = torch.ones(LAYER_NEURONS, dtype=torch.float32, device=device)

c_x_w = torch.randn(EMBEDDING_SIZE, LAYER_NEURONS, dtype=torch.float32, device=device)
c_h_w = torch.randn(LAYER_NEURONS, LAYER_NEURONS, dtype=torch.float32, device=device)
c_b = torch.zeros(LAYER_NEURONS, dtype=torch.float32, device=device)

i_x_w = torch.randn(EMBEDDING_SIZE, LAYER_NEURONS, dtype=torch.float32, device=device)
i_h_w = torch.randn(LAYER_NEURONS, LAYER_NEURONS, dtype=torch.float32, device=device)
i_b = torch.zeros(LAYER_NEURONS, dtype=torch.float32, device=device)

o_x_w = torch.randn(EMBEDDING_SIZE, LAYER_NEURONS, dtype=torch.float32, device=device)
o_h_w = torch.randn(LAYER_NEURONS, LAYER_NEURONS, dtype=torch.float32, device=device)
o_b = torch.zeros(LAYER_NEURONS, dtype=torch.float32, device=device)

output_w = torch.randn(LAYER_NEURONS, len(sentiment_labels), dtype=torch.float32, device=device)
output_b = torch.zeros(len(sentiment_labels), dtype=torch.float32, device=device)


parameters = [embedding_vector, f_x_w, f_h_w, f_b, c_x_w, c_h_w, c_b, i_x_w, i_h_w, i_b, o_x_w, o_h_w, o_b, output_w, output_b]

torch.nn.init.xavier_uniform_(f_x_w)
torch.nn.init.xavier_uniform_(f_h_w)
torch.nn.init.xavier_uniform_(c_x_w)
torch.nn.init.xavier_uniform_(c_h_w)
torch.nn.init.xavier_uniform_(i_x_w)
torch.nn.init.xavier_uniform_(i_h_w)
torch.nn.init.xavier_uniform_(o_x_w)
torch.nn.init.xavier_uniform_(o_h_w)

for p in parameters:
    p.requires_grad = True


optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(sentiment_labels)).to(device)
mean = torchmetrics.MeanMetric().to(device)

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch+1}")

    mean.reset()
    accuracy.reset()

    for b, (x, y) in enumerate(train_loader):
        x: torch.Tensor
        y: torch.Tensor

        x = x.to(device).permute(1, 0)
        y = y.to(device)

        embedded_x = embedding_vector[x]

        seq_size, b_size, emb_size = embedded_x.shape

        ct = torch.zeros(b_size, LAYER_NEURONS, dtype=torch.float32, device=device)
        h = torch.zeros(b_size, LAYER_NEURONS, dtype=torch.float32, device=device)

        mask = (x != review_tokens_stoi[PADDING_TOKEN]).float()

        for t in range(seq_size):
            f_gate = torch.sigmoid(embedded_x[t] @ f_x_w + h @ f_h_w + f_b)
            c_gate = torch.tanh(embedded_x[t] @ c_x_w + h @ c_h_w + c_b)
            i_gate = torch.sigmoid(embedded_x[t] @ i_x_w + h @ i_h_w + i_b)
            o_gate = torch.sigmoid(embedded_x[t] @ o_x_w + h @ o_h_w + o_b)

            mask_t = mask[t].unsqueeze(1)

            ct_new = f_gate * ct + i_gate * c_gate
            ct = mask_t * ct_new + (1 - mask_t) * ct

            h_new = o_gate * torch.tanh(ct)
            h = mask_t * h_new + (1 - mask_t) * h

        o = h @ output_w + output_b

        loss: torch.Tensor = criterion(o, y)
        optimizer.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            embedding_vector.grad[review_tokens_stoi[PADDING_TOKEN]].zero_()

        optimizer.step()

        mean.update(loss)
        accuracy.update(o, y)

        print(f"\rtrain_loss: {mean.compute().item():.4f}\t Batch: {b+1}/{len(train_loader)}", end="")

    print(f"\rtrain_loss: {mean.compute().item():.4f}\ttrain_acc: {accuracy.compute().item():.4f}")

    with torch.no_grad():
        mean.reset()
        accuracy.reset()

        for x, y in val_loader:
            x: torch.Tensor
            y: torch.Tensor

            x = x.to(device).permute(1, 0)
            y = y.to(device)

            embedded_x = embedding_vector[x]

            seq_size, b_size, emb_size = embedded_x.shape

            ct = torch.zeros(b_size, LAYER_NEURONS, dtype=torch.float32, device=device)
            h = torch.zeros(b_size, LAYER_NEURONS, dtype=torch.float32, device=device)

            mask = (x != review_tokens_stoi[PADDING_TOKEN]).float()

            for t in range(seq_size):
                f_gate = torch.sigmoid(embedded_x[t] @ f_x_w + h @ f_h_w + f_b)
                c_gate = torch.tanh(embedded_x[t] @ c_x_w + h @ c_h_w + c_b)
                i_gate = torch.sigmoid(embedded_x[t] @ i_x_w + h @ i_h_w + i_b)
                o_gate = torch.sigmoid(embedded_x[t] @ o_x_w + h @ o_h_w + o_b)

                mask_t = mask[t].unsqueeze(1)

                ct_new = f_gate * ct + i_gate * c_gate
                ct = mask_t * ct_new + (1 - mask_t) * ct

                h_new = o_gate * torch.tanh(ct)
                h = mask_t * h_new + (1 - mask_t) * h

            o = h @ output_w + output_b

            loss: torch.Tensor = criterion(o, y)

            mean.update(loss)
            accuracy.update(o, y)

        print(f"val_loss: {mean.compute().item():.4f}\tval_acc: {accuracy.compute().item():.4f}\n")

state_dict = {
    "embedding_vector": embedding_vector,
    "f_x_w": f_x_w,
    "f_h_w": f_h_w,
    "f_b": f_b,
    "c_x_w": c_x_w,
    "c_h_w": c_h_w,
    "c_b": c_b,
    "i_x_w": i_x_w,
    "i_h_w": i_h_w,
    "i_b": i_b,
    "o_x_w": o_x_w,
    "o_h_w": o_h_w,
    "o_b": o_b,
    "output_w": output_w,
    "output_b": output_b
}

torch.save(state_dict, MODEL_NAME)

with torch.no_grad():
    mean.reset()
    accuracy.reset()

    for x, y in test_loader:
        x: torch.Tensor
        y: torch.Tensor

        x = x.to(device).permute(1, 0)
        y = y.to(device)

        embedded_x = embedding_vector[x]

        seq_size, b_size, emb_size = embedded_x.shape

        ct = torch.zeros(b_size, LAYER_NEURONS, dtype=torch.float32, device=device)
        h = torch.zeros(b_size, LAYER_NEURONS, dtype=torch.float32, device=device)

        mask = (x != review_tokens_stoi[PADDING_TOKEN]).float()

        for t in range(seq_size):
            f_gate = torch.sigmoid(embedded_x[t] @ f_x_w + h @ f_h_w + f_b)
            c_gate = torch.tanh(embedded_x[t] @ c_x_w + h @ c_h_w + c_b)
            i_gate = torch.sigmoid(embedded_x[t] @ i_x_w + h @ i_h_w + i_b)
            o_gate = torch.sigmoid(embedded_x[t] @ o_x_w + h @ o_h_w + o_b)

            mask_t = mask[t].unsqueeze(1)

            ct_new = f_gate * ct + i_gate * c_gate
            ct = mask_t * ct_new + (1 - mask_t) * ct

            h_new = o_gate * torch.tanh(ct)
            h = mask_t * h_new + (1 - mask_t) * h

        o = h @ output_w + output_b

        loss: torch.Tensor = criterion(o, y)

        mean.update(loss)
        accuracy.update(o, y)

    print(f"test_loss: {mean.compute().item():.4f}\ttest_acc: {accuracy.compute().item():.4f}\n")