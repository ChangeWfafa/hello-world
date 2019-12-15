import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(    # input_size: (1,28,28)
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

        )# (16,14,14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

        )# (32,7,7)
        self.out = nn.Linear(32*7*7,128)
        self.lin = nn.Linear(128,10)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,32*7*7)
        x = self.out(x)
        x = self.lin(x)
        return x