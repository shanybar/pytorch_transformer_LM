import time
import copy
import torch
from torch import nn
from src.data_processing import load_data
from src.models import TransformerLM

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, val_data, test_data, vocab = load_data()
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = val_data.to(device)

    num_tokens = len(vocab) # size of vocabulary
    embed_size = 200 # embedding dimension
    d_hid = 200 # dimension of the feedforward network model in nn.TransformerEncoder
    num_layers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    num_head = 2 # number of heads in nn.MultiheadAttention
    dropout = 0.2
    model = TransformerLM(num_tokens, embed_size, num_head, d_hid, num_layers, dropout).to(device)

    loss_func = nn.CrossEntropyLoss()
    lr = 0.5
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1)


def train_single_epoch(model: nn.Module, optimizer, scheduler, loss_func):
    pass


def eval_model():
    pass