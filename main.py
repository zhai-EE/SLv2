from DataSet import Cifar10_train
from torch.utils.data import DataLoader
from Net import MyCnn
from torch import nn, optim
import torch
from Eval import eval, eval_ds
import matplotlib.pyplot as plt
from MyLoss import SL, LSRLoss
from MyResNet import resnetDs, resnet50
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import InterpolationMode

LossClass = None  # 使用哪个loss函数
addNoise = None
runName = None  # 根据loss函数和是否加噪声生成一个字符串标识数据


def showRes(name, save=False):
    acc = torch.load(runName + "_result.pt")
    plotAcc(acc, save=save, name=name)


def showResDs(name, save=False):
    acc = torch.load(runName + "_result.pt")
    plotAccDs(acc, name=name, save=save)


def plotAcc(acc, save=False, name=runName):
    fig, ax = plt.subplots()
    numEpoch = acc.shape[0]
    xtick = torch.linspace(0, numEpoch - 1, numEpoch)
    overall = torch.mean(acc, dim=1)
    max = acc.max(dim=1).values
    min = acc.min(dim=1).values
    # 每隔5个抽出
    gap = 5
    for i in range(acc.shape[1]):
        ax.plot(xtick[::gap], acc[:, i][::gap], label=f"class {i}", linestyle='dashed')
    ax.plot(xtick[::gap], overall[::gap], label=f"overall", linestyle='solid', linewidth=3,
            color=(196 / 256, 78 / 256, 82 / 256))
    ax.fill_between(xtick[::gap], min[::gap], max[::gap], color=(230 / 256, 230 / 256, 250 / 256))
    ax.set_xlabel("epoch")
    ax.set_ylabel("class wise accuracy")
    ax.set_title(name)
    # ax.legend(ncol=2, loc='upper right')
    ax.grid()
    plt.ylim((0.8, 1))
    if save:
        plt.savefig(name + ".png")
    plt.show()


def plotAccDs(acc, save=False, name=runName):
    layerName = ['conv2_x', 'conv3_x', 'conv4_x', 'conv5_x', 'final']
    for i in range(5):
        plotAcc(acc[:, i, :], save=save, name=f"{name} at {layerName[i]}")


def train_ds():
    batchsize = 64
    lr = 0.01
    numEpoch = 120
    accuracy = torch.zeros(numEpoch, 5, 10, dtype=torch.float32)
    #
    device = torch.device('cuda')
    #
    dataset = Cifar10_train(addNoise)
    dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=2)
    model = resnetDs().to(device, dtype=torch.float32)
    #
    optimizer = optim.SGD(model.parameters(), weight_decay=1e-4, lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1)
    criterion = LossClass()
    for epoch in range(numEpoch):
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            #
            pred2, pred3, pred4, pred5, predFinal = model(x)
            loss = criterion(pred2, y) + criterion(pred3, y) + criterion(pred4, y) + criterion(pred5, y) + criterion(
                predFinal, y)
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"epoch:{epoch}\t batch:{i} \tloss:{loss}")
        scheduler.step()
        infos = eval_ds(model)
        #
        for j in range(10):
            for k in range(5):
                accuracy[epoch, k, j] = infos[k][j].accuracy
                if epoch % 5 == 0:
                    print(f"epoch:{epoch}\tlayer:{k}\tclass:{j}\taccuracy:{infos[k][j].accuracy}")
    torch.save(model, runName + ".pt")
    torch.save(accuracy, runName + "_result.pt")
    plotAccDs(accuracy, save=True)


def train():
    batchsize = 256
    lr = 0.01
    numEpoch = 120
    accuracy = torch.zeros(numEpoch, 10, dtype=torch.float32)
    #
    device = torch.device('cuda')
    #
    dataset = Cifar10_train(addNoise)
    dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=2)
    model = resnet50().to(device, dtype=torch.float32)
    #
    optimizer = optim.SGD(model.parameters(), weight_decay=1e-4, lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1)
    criterion = LossClass()
    for epoch in range(numEpoch):
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            #
            pred = model(x)
            loss = criterion(pred, y)
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"epoch:{epoch}\t batch:{i} \tloss:{loss}")
        scheduler.step()
        infos = eval(model)
        #
        for j in range(10):
            accuracy[epoch, j] = infos[j].accuracy
            if epoch % 5 == 0:
                print(f"epoch:{epoch}\tclass:{j}\taccuracy:{infos[j].accuracy}")
    torch.save(model, runName + ".pt")
    torch.save(accuracy, runName + "_result.pt")
    plotAcc(accuracy, save=True, name=runName)


if __name__ == '__main__':
    #
    if (True):
        LossClass = nn.CrossEntropyLoss
        addNoise = True
        runName = f"{LossClass()._get_name()}  isNoisy_{addNoise}"
        showRes("CE - noisy", save=True)
    else:
        #
        addNoise = False
        LossClass = nn.CrossEntropyLoss
        runName = f"{LossClass()._get_name()}  isNoisy_{addNoise}"
        train()
        addNoise = True
        for LossClass in [nn.CrossEntropyLoss, SL, LSRLoss]:
            runName = f"{LossClass()._get_name()}  isNoisy_{addNoise}"
            train()
