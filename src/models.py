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
        pass