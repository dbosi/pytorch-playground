import numpy as np
import torch
import re
import pandas
import pathlib
import kagglehub


### CONSTANTS ###

MODEL_NAME = pathlib.Path("model.pth")

CPU_DEVICE = "cpu"
CUDA_DEVICE = "cuda"

DATASET_HANDLE = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
DATASET_NAME = "IMDB Dataset.csv"

REGEX_STRING = r"(<br />|\"|\,|\.|!|\?|\(|\)|;|:|\\|\*|-|/)"

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

reviews_clean = reviews.str.replace(REGEX_STRING, " ", regex=True)


reviews_tokens_lists = reviews_clean.apply(lambda x: x.lower().split())
reviews_tokens = sorted(set(np.concatenate(reviews_tokens_lists.to_numpy())))
reviews_tokens.insert(0, PADDING_TOKEN)
reviews_tokens.insert(1, UNKNOWN_TOKEN)

review_tokens_stoi = {str(word): idx for idx, word in enumerate(reviews_tokens)}
review_tokens_itos = {idx: word for word, idx in review_tokens_stoi.items()}


sentiment_labels = sorted(dataframe[sentiment_col].unique().tolist())
sentiment_labels_stoi = {str(word): idx for idx, word in enumerate(sentiment_labels)}
sentiment_labels_itos = {idx: word for word, idx in sentiment_labels_stoi.items()}


### TEST DATA ###

positive_review_1 = "I really liked this movie!!!"
positive_review_2 = "The movie is good."

negative_review_1 = "I didn't like this movie!!!"
negative_review_2 = "It's bad!!!"

list_of_reviews = [positive_review_1, positive_review_2, negative_review_1, negative_review_2]

list_of_review_tokens = [re.sub(REGEX_STRING, " ", review).lower().split() for review in list_of_reviews]
list_of_review_tokens_idx = [[review_tokens_stoi.get(token, review_tokens_stoi[UNKNOWN_TOKEN]) for token in review] for review in list_of_review_tokens]

max_len = max([len(review) for review in list_of_review_tokens_idx])

padded_list_of_review_tokens_idx = [review + [review_tokens_stoi[PADDING_TOKEN]] * (max_len - len(review)) for review in list_of_review_tokens_idx]

x = torch.tensor(padded_list_of_review_tokens_idx, dtype=torch.long, device=device)


### MODEL ###

state_dict = torch.load(MODEL_NAME)

embedding_vector = state_dict["embedding_vector"]

f_x_w = state_dict["f_x_w"].to(device)
f_h_w = state_dict["f_h_w"].to(device)
f_b = state_dict["f_b"].to(device)

c_x_w = state_dict["c_x_w"].to(device)
c_h_w = state_dict["c_h_w"].to(device)
c_b = state_dict["c_b"].to(device)

i_x_w = state_dict["i_x_w"].to(device)
i_h_w = state_dict["i_h_w"].to(device)
i_b = state_dict["i_b"].to(device)

o_x_w = state_dict["o_x_w"].to(device)
o_h_w = state_dict["o_h_w"].to(device)
o_b = state_dict["o_b"].to(device)

output_w = state_dict["output_w"].to(device)
output_b = state_dict["output_b"].to(device)

with torch.no_grad():
    x: torch.Tensor

    x = x.to(device).permute(1, 0)
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

    predictions = torch.softmax(o, dim=1).argmax(dim=1).detach().tolist()

    for i in range(len(list_of_reviews)):
        print(f"{list_of_reviews[i]} : {sentiment_labels_itos[predictions[i]]}")