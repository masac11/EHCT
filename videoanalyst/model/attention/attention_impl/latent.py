# --------------------------------------------------------
#  Temporal-Spatial Feature Fusion (TSFF)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from videoanalyst.model.attention.attention_base import (TRACK_ATTENTION, VOS_ATTENTION)
from videoanalyst.model.module_base import ModuleBase



@VOS_ATTENTION.register
@TRACK_ATTENTION.register
class Latent(ModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, f1, f2):
        return f1

     
