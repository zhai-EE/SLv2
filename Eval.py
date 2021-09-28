from DataSet import Cifar10_test
from torch.utils.data import DataLoader
from Net import MyCnn
from torch import nn, optim
import torch


class Info:
    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.accuracy = 0


def eval(model):
    with torch.no_grad():
        infos = []
        for i in range(10):
            infos.append(Info())
        batchsize = 128
        device = torch.device('cuda')
        dataset = Cifar10_test()
        datasetLen = len(dataset)
        dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=2)
        #
        model = model.to(device)
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            #
            pred = model(x)
            pred = pred.argmax(dim=1).to(torch.long)
            for j in range(len(pred)):
                if pred[j] == y[j]:
                    infos[y[j]].TP += 1
                else:
                    infos[pred[j]].FP += 1
                    infos[y[j]].FN += 1
        for i in range(10):
            infos[i].TN = datasetLen - infos[i].FP - infos[i].FN - infos[i].TP
            infos[i].accuracy = (infos[i].TN + infos[i].TP) / (infos[i].TN + infos[i].TP + infos[i].FP + infos[i].FN)

    return infos






