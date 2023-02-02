import math
import time
import copy
import torch
from torch import nn
from src.data_processing import load_data, get_batch
from src.models import TransformerLM, generate_mask_matrix


def train_model():
    print(torch.backends.cudnn.enabled)
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
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
        train_single_epoch(model, train_data, optimizer, scheduler, loss_func, bptt,
                           num_tokens, epoch, device)
        val_loss = eval_model(model,val_data,loss_func, bptt, num_tokens, device)
        val_perplexity = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch: {epoch:3d} | time: {elapsed:5.2f}s |'
              f' val loss {val_loss: 5.2f} | val perplexity {val_perplexity: 8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        scheduler.step()

    test_loss = eval_model(best_model, test_data, loss_func, bptt, num_tokens, device)
    test_perplexity = math.exp(test_loss)
    print('=' * 89)
    print(f'End of training | test loss: {test_loss: 5.2f} | '
          f'test perplexity: {test_perplexity: 8.2f}')
    print('=' * 89)


def train_single_epoch(model: nn.Module, train_data, optimizer, scheduler, loss_func,
                       bptt: int, num_tokens: int, epoch, device) -> None:
    model.train() # turn on train mode
    total_loss = .0
    log_interval = 200
    start_time = time.time()
    src_mask = generate_mask_matrix(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, labels = get_batch(train_data, i, bptt)
        seq_len = data.size(0)
        if seq_len != bptt: # only for the last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        output = output.view(-1, num_tokens)
        loss = loss_func(output, labels)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            last_lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            curr_loss = total_loss / log_interval
            perplexity = math.exp(curr_loss)
            print(f'| epoch {epoch:3d} | batch {batch:5d}/{num_batches:5d}')
            print(f'| lr {last_lr:02.2f} | ms/batch {ms_per_batch:5.2f}')
            print(f'| loss {curr_loss:5.2f} | perplexity {perplexity: 5.2f}')
            total_loss = 0
            start_time = time.time()


def eval_model(model: nn.Module, eval_data, loss_func,
               bptt: int, num_tokens: int, device) -> float:
    model.eval()
    total_loss = 0.
    src_mask = generate_mask_matrix(bptt).to(device)

    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, labels = get_batch(eval_data, i, bptt)
            seq_len = data.size(0)
            if seq_len != bptt:  # only for the last batch
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, num_tokens)
            total_loss += seq_len * loss_func(output_flat, labels).item()

    return total_loss / len(eval_data)

if __name__ == '__main__':
    train_model()