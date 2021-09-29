import torch.nn as nn


class MyCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((4 * 4) * 64, 8 * 64),
            nn.ReLU(),
            nn.Linear(8 * 64, 10),
        )

    def forward(self, x):
        return self.net(x)




