#识别率：87.43%
#训练次数：200
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lin = nn.Sequential(
            nn.Linear(28*28,16*16),
            nn.ReLU(),
            nn.Linear(16*16,8*8),
            nn.ReLU(),
            nn.Linear(8*8,10)
        )


    def forward(self,x):
        x = self.lin(x)
        return x