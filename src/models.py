import math
from typing import Tuple

import torch
from torch import nn
import torch.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class TransformerLM(nn.Module):