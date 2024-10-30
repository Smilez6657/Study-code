import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


# def main():
transform = transforms.Compose(       # 预处理
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
         # normalize使用均值和标准差来标准化函数
    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
            # 数据集下载
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,       # 随机拿出36张图片
                                               shuffle=True, num_workers=0)  # shuffle 表示是否打乱顺序
#
#     # 10000张测试图片
#     # 第一次使用时要将download设置为True才会自动去下载数据集
val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)
val_data_iter = iter(val_loader)
val_image, val_label = next(val_data_iter)
val_image = val_image.to(device)
val_label = val_label.to(device)

classes = ('plane', 'car', 'bird', 'cat',       # 标签值
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

### 下面的这部分代码用来检验代码，显示四张图片 ###
            # def imshow(img):
            #     img = img / 2 + 0.5     # unnormalize 反标准化处理
            #     npimg = img.numpy()     # 转化为numpy格式
            #     plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 交换从c,h,w的顺序为h,w,c
            #     plt.show()
            #
            #     # print labels
            # print(' '.join('%5s' % classes[val_label[j]] for j in range(4)))
            #     # show images
            # imshow(torchvision.utils.make_grid(val_image))


net = LeNet()
net.to(device)

loss_function = nn.CrossEntropyLoss()     # 损失函数
loss_function.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)   # adam优化器，前半句意为将net中可训练的参数全部进行训练  learning rate 为学习率



for epoch in range(5):  # 迭代五次，loop over the dataset multiple times

    running_loss = 0.0  # 累加训练过程中的损失
    for step, data in enumerate(train_loader, start=0):  # 该循环用来遍历训练集的样本
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()   # 清零历史损失梯度
        # forward + backward + optimize
        outputs = net(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:  # print every 500 mini-batches
            with torch.no_grad():  # 上下文管理器
                outputs = net(val_image)  # [batch, 10]

                predict_y = torch.max(outputs, dim=1)[1]  # 找出最可能的index
                accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                        (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')

save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)  # 保存模型

#
#
# if __name__ == '__main__':
#     main()