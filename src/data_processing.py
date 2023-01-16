
import torch
from torch import Tensor
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def process_data(tokenizer, vocab, raw_text_iter: dataset.IterableDataset) -> Tensor:
    """
    Converts raw text into a flat tensor
    """
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, batch_size: int) -> Tensor:
    """
    Divides the data into batch_size separate elements, removing extra elements
    that would not fit.

    :param data: Tensor, shape [N]
    :param batch_size: int, batch size
    :return: Tensor of shape [N // batch_size, batch_size]
    """
    seq_len = data.size() // batch_size


def load_data():
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])


