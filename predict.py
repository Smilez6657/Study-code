
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet



transform = transforms.Compose(
        [transforms.Resize((32, 32)),   # 缩放用来预测的图片
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))

im = Image.open('1.jpg')
im = transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # 增加一个新的维度，[N, C, H, W]

with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
print(classes[int(predict[0])])
#  也可使用softmax来预测，会输出每个类的概率，下面为代码
#  predict = torch.softmax(outputs, dim=1)
#  print(predict)




