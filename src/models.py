import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, heads_num: int, d_hid: int,
                 layers_num: int, dropout:float=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.position_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layers = TransformerEncoderLayer(d_model, heads_num, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, layers_num)
        self.embed_encoder = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.linear_decoder = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self) -> None:
        init_range = 0.1
        self.embed_encoder.weight.data.uniform_(-init_range, init_range)
        self.linear_decoder.bias.data.zero_()
        self.linear_decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """

        :param src: Tensor, shape [seq_len, batch_size]
        :param src_mask: Tensor, shape [seq_len, seq_len]

        :return: output Tensor of shape [seq_len, batch_size, vocab_size]
        """
        src = self.embed_encoder(src)
        src *= math.sqrt(self.d_model)
        src = self.position_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear_decoder(output)
        return output


def generate_mask_matrix(size: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    :param size: matrix size
    :return: mask tensor
    """
    return torch.triu(torch.ones(size,size) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    pass