import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU():
    def forward(self, x):
        return F.sigmoid(x) * x