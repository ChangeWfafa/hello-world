#识别率：82.3%
#训练次数：100
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(28*28,10)
    def forward(self,x):
        x = self.lin1(x)

        return x