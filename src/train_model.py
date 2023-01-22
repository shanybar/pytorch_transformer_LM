import time
import copy
import torch
from torch import nn
from src.data_processing import load_data, get_batch
from src.models import TransformerLM, generate_mask_matrix

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
    bptt = 35
    model = TransformerLM(num_tokens, embed_size, num_head, d_hid, num_layers, dropout).to(device)

    loss_func = nn.CrossEntropyLoss()
    lr = 0.5
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)

    best_val_loss = float('-inf')
    epochs = 10
    best_model = None

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train_single_epoch(model, train_data, optimizer, scheduler, loss_func, bptt, num_tokens)


def train_single_epoch(model: nn.Module, train_data, optimizer, scheduler, loss_func,
                       bptt: int, num_tokens: int) -> None:
    model.train() # turn on train mode
    total_loss = .0
    log_interval = 100
    start_time = time.time()
    src_mask = generate_mask_matrix(bptt)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1), bptt):
        data, labels = get_batch(train_data, i, bptt)
        seq_len = data.size(0)
        if seq_len != bptt: # only for the last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = loss_func(output.view(-1, num_tokens), labels)

        optimizer.zero_grad()
        loss.backward()



def eval_model():
    pass