from DataSet import Cifar10_test
from torch.utils.data import DataLoader
from Net import MyCnn
from torch import nn, optim
import torch


class Info:
    # 用来存储每一个类的统计信息
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
                # 预测正确则对应类的TP数++
                if pred[j] == y[j]:
                    infos[y[j]].TP += 1
                # 预测错误的话，标签对应类的FN和预测对应类的FP++
                else:
                    infos[pred[j]].FP += 1
                    infos[y[j]].FN += 1

        for i in range(10):
            # TP+TN+FP+FN应该等于数据集的样本数
            infos[i].TN = datasetLen - infos[i].FP - infos[i].FN - infos[i].TP
            infos[i].accuracy = (infos[i].TN + infos[i].TP) / (infos[i].TN + infos[i].TP + infos[i].FP + infos[i].FN)

    return infos


def eval_ds(model):
    with torch.no_grad():
        # 初始化info矩阵5行10列，行下标表示resnet各层的输出
        infos = []
        for j in range(5):
            infos_ = []
            for i in range(10):
                infos_.append(Info())
            infos.append(infos_)
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
            pred2, pred3, pred4, pred5, predFinal = model(x)
            results = [pred2, pred3, pred4, pred5, predFinal]
            for k in range(5):
                pred = results[k]
                pred = pred.argmax(dim=1).to(torch.long)
                for j in range(len(pred)):
                    # 预测正确则对应类的TP数++
                    if pred[j] == y[j]:
                        infos[k][y[j]].TP += 1
                    # 预测错误的话，标签对应类的FN和预测对应类的FP++
                    else:
                        infos[k][pred[j]].FP += 1
                        infos[k][y[j]].FN += 1

        for k in range(5):
            for i in range(10):
                # TP+TN+FP+FN应该等于数据集的样本数
                infos[k][i].TN = datasetLen - infos[k][i].FP - infos[k][i].FN - infos[k][i].TP
                infos[k][i].accuracy = (infos[k][i].TN + infos[k][i].TP) / (
                        infos[k][i].TN + infos[k][i].TP + infos[k][i].FP + infos[k][i].FN)

    return infos
