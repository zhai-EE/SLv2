import torch
from torch import nn


class RCE(nn.Module):
    def __init__(self):
        super.__init__()
        self.A = -6

    def forward(self, x, y):
        py = x[:, y]
        return torch.mean(-self.A * (1 - py))


class SL(nn.Module):
    def __init__(self):
        super.__init__()
        self.CE = nn.CrossEntropyLoss()
        self.RCE = RCE()
        self.alpha = 0.1
        self.beta = 1

    def foward(self, x, y):
        return self.alpha * self.CE(x, y) + self.beta * self.RCE(x, y)
