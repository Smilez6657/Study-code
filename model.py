import torch.nn as nn
import torch.nn.functional as F
from humanfriendly.terminal import output
import torch


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()   # super解决调用nn过程中可能出现的问题
        self.conv1 = nn.Conv2d(3, 16, 5)  # 定义第一个卷积层
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
                                     # F.relu 是 PyTorch 中的一个激活函数，表示应用 ReLU（Rectified Linear Unit）激活函数。
                                     # ReLU 的数学表达式为f(x)=max(0,x)，即当输入为正时输出其值，当输入为负时输出0。
                                     # 这种非线性函数有助于引入非线性特征，使神经网络能够学习更复杂的映射。
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x


