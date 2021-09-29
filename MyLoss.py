import torch
from torch import nn
import torch.nn.functional as F


class RCE(nn.Module):
    def __init__(self):
        super(RCE, self).__init__()
        self.A = torch.tensor(-6)

    def forward(self, x, y):
        x = F.softmax(x, dim=1)
        py = x[:, y]
        return torch.mean(-self.A * (1 - py))


class SL(nn.Module):
    def __init__(self):
        super(SL, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.RCE = RCE()
        self.alpha = 0.1
        self.beta = 1

    def forward(self, x, y):
        return self.alpha * self.CE(x, y) + self.beta * self.RCE(x, y)


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, alpha, numClass):
        assert 0.0 < alpha <= 1.0

        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = alpha / numClass
        one_hot = torch.full((numClass,), smoothing_value)

        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - alpha

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)  # 扩展为矩阵,值全为alpha/K
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)  # 正确标签处设为1-alpha

        return F.kl_div(output, model_prob, reduction='sum')
