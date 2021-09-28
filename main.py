from DataSet import Cifar10_train
from torch.utils.data import DataLoader
from Net import MyCnn
from torch import nn, optim
import torch
from Eval import eval
import matplotlib.pyplot as plt


def plotAcc(acc):
    fig, ax = plt.subplots()
    numEpoch = acc.shape[0]
    xtick = torch.linspace(0, numEpoch - 1, numEpoch)
    for i in acc.shape[0]:
        ax.plot(xtick, acc[:, i], label=f"class {i}")
    ax.set_xlabel("epoch")
    ax.set_xlabel("class wise accuracy")

if __name__ == '__main__':
    #
    batchsize = 512
    lr = 0.01
    numEpoch = 120
    accuracy = torch.zeros(numEpoch, 10, dtype=torch.float32)
    #
    device = torch.device('cuda')
    #
    dataset = Cifar10_train()
    dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=2)
    model = MyCnn().to(device, dtype=torch.float32)
    optimizer = optim.SGD(model.parameters(), weight_decay=1e-4, lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
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
        for j in range(10):
            accuracy[epoch, j] = infos[j].accuracy
            if epoch % 5 == 0:
                print(f"epoch:{epoch}\tclass{j}\taccuracy:{infos[j].accuracy}")
