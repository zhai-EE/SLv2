import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random


def showImage(imgTensor):
    imgTensor = torch.permute(imgTensor, [1, 2, 0])
    plt.imshow(imgTensor / 255)
    plt.show()


def readBatch(i):
    import pickle
    fileName = "./cifar-10-batches-py/data_batch_" + str(i)
    with open(fileName, 'rb') as f:
        dataDict = pickle.load(f, encoding='bytes')
    return dataDict


def readMetaData():
    import pickle
    fileName = "./cifar-10-batches-py/batches.meta"
    with open(fileName, 'rb') as f:
        dataDict = pickle.load(f, encoding='bytes')
    return dataDict


def fechDataFromFile():
    datas = []
    labels = []
    for i in range(1, 6):
        dataDict = readBatch(i)
        datas.append(
            torch.permute(
                torch.as_tensor(dataDict[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1),
                                dtype=torch.float32),
                [0, 3, 1, 2]
            )
        )
        labels.append(torch.as_tensor(dataDict[b'labels'], dtype=torch.long))
    x = torch.cat(datas, 0)
    y = torch.cat(labels, 0)
    return x, y


def addSymmetricNoise(labels, noiseRate):
    ids = range(len(labels))
    idToFlip = random.sample(ids, int(noiseRate * len(labels)))
    for id in idToFlip:
        newLabel = random.randrange(10)
        while newLabel == labels[id]:
            newLabel = random.randrange(10)
        labels[id] = newLabel
    return labels


class Cifar10_train(Dataset):
    def __init__(self, addNoise):
        self.x, self.y = fechDataFromFile()
        self.metadata = readMetaData()
        # 40%标签随机翻转
        if addNoise:
            self.y = addSymmetricNoise(self.y, 0.4)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Cifar10_test(Dataset):
    def __init__(self):
        import pickle
        fileName = "./cifar-10-batches-py/test_batch"
        with open(fileName, 'rb') as f:
            dataDict = pickle.load(f, encoding='bytes')
        self.x = torch.permute(
            torch.as_tensor(dataDict[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1),
                            dtype=torch.float32),
            [0, 3, 1, 2]
        )
        self.y = torch.as_tensor(dataDict[b'labels'], dtype=torch.long)
        # showImage(self.x[15])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
