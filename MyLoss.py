import torch
from torch import nn
import torch.nn.functional as F


class RCE(nn.Module):
    def __init__(self):
        super(RCE, self).__init__()
        self.A = torch.tensor(-6)

    def forward(self, x, y):
        x = F.softmax(x, dim=1)
        py = x[:, y]  # 取出在正确标签上的概率值
        return torch.mean(-self.A * (1 - py))  # 这里使用的是化简后的RCE表达式，原文有证


class SL(nn.Module):
    def __init__(self):
        super(SL, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.RCE = RCE()
        self.alpha = 0.1
        self.beta = 1

    def forward(self, x, y):
        return self.alpha * self.CE(x, y) + self.beta * self.RCE(x, y)


class LSRLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, alpha=0.1, numClass=10):
        assert 0.0 < alpha <= 1.0

        super(LSRLoss, self).__init__()

        smoothing_value = alpha / numClass
        one_hot = torch.full((numClass,), smoothing_value)

        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - alpha

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        output = F.log_softmax(output, dim=1)  # log(softmax(x))
        smoothedLabel = self.one_hot.repeat(target.size(0), 1).to(output.device)  # 扩展为矩阵,值全为alpha/K
        smoothedLabel = smoothedLabel.scatter_(1, target.unsqueeze(1), self.confidence)  # 正确标签处设为1-alpha
        return torch.mean(torch.sum(-output * smoothedLabel, dim=1))  # 跟log(softmax(x))配合起来就是交叉熵
