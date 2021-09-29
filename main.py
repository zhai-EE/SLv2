from DataSet import Cifar10_train
from torch.utils.data import DataLoader
from Net import MyCnn
from torch import nn, optim
import torch
from Eval import eval
import matplotlib.pyplot as plt
from MyLoss import SL
from MyResNet import resnetDs
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import InterpolationMode

LossClass = nn.CrossEntropyLoss
addNoise = False
runName = f"{LossClass()._get_name()}  isNoisy_{addNoise}"


def showRes(name):
    acc = torch.load(name + "_result.pt")
    plotAcc(acc)


def plotAcc(acc, save=False):
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
    ax.set_title(runName)
    ax.legend()
    ax.grid()
    plt.ylim((0.5, 1))
    if save:
        plt.savefig(runName + ".png")
    plt.show()


if __name__ == '__main__':
    #
    if (False):
        showRes(runName)
    else:
        batchsize = 64
        lr = 0.01
        numEpoch = 120
        accuracy = torch.zeros(numEpoch, 10, dtype=torch.float32)
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
                # x = resize(x, [64, 64], interpolation=InterpolationMode.BILINEAR, antialias=True)
                y = y.to(device)
                #
                pred2, pred3, pred4, pred5, predFinal = model(x)
                loss = criterion(pred2, y) + criterion(pred3, y) + criterion(pred4, y) + criterion(pred5, y) + criterion(predFinal, y)
                #
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f"epoch:{epoch}\t batch:{i} \tloss:{loss}")
            scheduler.step()
            infos = eval(model)
            for j in range(10):
                accuracy[epoch, j] = infos[j].accuracy
                if epoch % 5 == 0:
                    print(f"epoch:{epoch}\tclass{j}\taccuracy:{infos[j].accuracy}")
        torch.save(model, runName + ".pt")
        torch.save(accuracy, runName + "_result.pt")
        plotAcc(accuracy, save=True)
