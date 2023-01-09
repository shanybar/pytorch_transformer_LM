import math
from typing import Tuple

import torch
from torch import nn
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




class PositionalEncoding(nn.Module):
    pass