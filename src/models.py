import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
    def __init__(self, d_model: int, dropout: float = 0.1, max_len:int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000.0)/d_model))
        position_embeds = torch.zeros(max_len, 1, d_model)
        position_embeds[:, 0, 0::2] = torch.sin(position * div_term)
        position_embeds[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_embeds', position_embeds) # Saved to state_dict
        # but stayed fixed and won't be updated by the optimizer

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Tensor, shape [seq_len, batch_size, embeddings_dim]
        :return: the tensor x summed with the position embeddings
        """
        x = x + self.position_embeds[:x.size(0)]
        return self.dropout(x)
