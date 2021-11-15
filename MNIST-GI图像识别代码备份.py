import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_size = 28  # 128
iteration = 128  # 12


# field = np.random.normal(0, 1, [iteration, image_size, image_size])
# field = torch.from_numpy(field).cuda()

class MY_MNIST(Dataset):

    def __init__(self, root, flag, transform=None):
        self.transform = transform
        pre_data, self.targets = torch.load(root)

        xxx = 0
        if flag == 0:
            xxx = 600  # 60000
            batch_size = 1  # 1
            self.data = torch.zeros((600, iteration, image_size, image_size))  # 60000
        else:
            xxx = 1  # 1
            batch_size = 100  # 10000
            self.data = torch.zeros((100, iteration, image_size, image_size))  # 10000

        for num in range(xxx):
            if num % 5000 == 0:
                print("Num = %d" % (num + 1))
            torch.cuda.empty_cache()

            # 初始化 B 数组
            B = torch.zeros((iteration, batch_size)).cuda()
            BI = torch.zeros((batch_size, iteration, image_size, image_size)).cuda()

            # [500, 28, 28]的图像数组
            img_batch = torch.zeros((batch_size, image_size, image_size))
            # 读取图片
            for i in range(batch_size):
                x = pre_data[i + batch_size * num]
                transform1 = transforms.ToPILImage(mode="L")
                y = transform1(np.uint8(x.numpy()))
                y = y.resize((image_size, image_size))
                transform2 = transforms.ToTensor()
                z = transform2(y)
                img_batch[i] = z[0]
            img_batch = img_batch.cuda()

            field = np.random.normal(0, 1, [iteration, image_size, image_size])
            field = torch.from_numpy(field).cuda()

            for k in range(iteration):
                temp = field[k].repeat(batch_size, 1, 1).cuda()
                B[k] = torch.mul(temp, img_batch).sum(2).sum(1)
            B = B.permute(1, 0).cpu()

            for i in range(batch_size):
                for j in range(iteration):
                    BI[i][j] = B[i][j] * field[j]
                self.data[i + batch_size * num] = BI[i]

    #             for i in range(batch_size):
    #                 x = img_batch[i]
    #                 self.data[i + batch_size * num][0] = x

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def __len__(self):
        return len(self.data)


train_dataset = MY_MNIST(root='C:/Users/chenda/Desktop/代码/data/MNIST/processed/training.pt', flag=0)
test_dataset = MY_MNIST(root='C:/Users/chenda/Desktop/代码/data/MNIST/processed/test.pt', flag=1)

print('done')
plt.imshow(train_dataset[0][0][0])


class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


print('done')

torch.cuda.empty_cache()
batch_size = 4  # 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(128, 256, 3, padding=1)  # 改动了！！！
#         self.conv2 = nn.Conv2d(256, 512, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#
#         self.bn1 = nn.BatchNorm2d(256)
#         self.bn2 = nn.BatchNorm2d(512)
#
#         self.relu = nn.ReLU()
#         self.res = ResidualBlock(256)
#
#         self.drop = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(512 * 7 * 7, 512)
#         self.fc2 = nn.Linear(512, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#
#         x = self.res(x)
#         x = self.res(x)
#         x = self.res(x)
#         x = self.res(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.pool(x)
#
#         #         print(" x shape ",x.size())
#         x = x.view(-1, 512 * 7 * 7)
#         x = F.relu(self.fc1(x))
#         x = self.drop(x)
#         x = self.fc2(x)
#         return x


class DoubleConv(nn.Module):  # 封装常用的两次卷积操作
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        input = self.conv(input)
        return input


class UpSample(nn.Module):  # 封装上采样加卷积操作
    def __init__(self, in_ch, out_ch):
        super(UpSample, self).__init__()
        self.sample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, (1, 1)),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, input):
        input = self.sample(input)
        return input


class Net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Net, self).__init__()
        # 下采样
        # self.conv1 = DoubleConv(in_ch, 64)
        # self.pool1 = nn.MaxPool2d(2)
        # self.conv2 = DoubleConv(64, 128)
        # self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(in_ch, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        # 上采样
        self.up6 = UpSample(1024, 512)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = UpSample(512, 256)
        self.conv7 = DoubleConv(512, 256)

        self.newup8 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up8 = UpSample(256, 128)
        self.conv8 = DoubleConv(256, 128)
        # self.up9 = UpSample(128, 64)
        self.newup9 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, (1, 1))

        self.Fla = nn.Flatten()  # 新加的
        self.Lin1 = nn.Linear(1 * 112 * 112, 784)  # 16个batch_size
        self.Lin2 = nn.Linear(784, 10)

    def forward(self, x):
        # c1 = self.conv1(x)
        # p1 = self.pool1(c1)
        # c2 = self.conv2(p1)
        # p2 = self.pool2(c2)
        c3 = self.conv3(x)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        u6 = self.up6(c5)
        merge6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(merge6)
        u7 = self.up7(c6)
        merge7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(merge7)
        # u8 = self.up8(c7)
        u8 = self.newup8(c7)
        # merge8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)
        # u9 = self.up9(c8)
        u9 = self.newup9(c8)
        # merge9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)
        c10 = self.conv10(c9)
        tmp_out = nn.Softsign()(c10)
        out = self.Fla(tmp_out)
        out = self.Lin1(out)
        out = self.Lin2(out)
        return out


model = Net(128, 1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00002, weight_decay=0.0003)

model = model.to(device)
criterion = criterion.to(device)


# 训练
def train(epoch):
    model.train()  # 作用是启用batch normalization和drop out
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()  # 把梯度置零

        data = data.to(device)
        target = target.to(device)

        output = model(data)
        # print(output.shape)
        # print(target.shape)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data.cuda()), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


# 测试集测试
def test():
    model.eval()  # 神经网络会沿用batch normalization的值，并不使用drop out
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)

        data = data.to(device)
        target = target.to(device)

        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data.item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


# 训练集测试
def test2():
    model.eval()  # 神经网络会沿用batch normalization的值，并不使用drop out
    test_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = Variable(data), Variable(target)

        data = data.to(device)
        target = target.to(device)

        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data.item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(train_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


for epoch in range(1, 40):
    train(epoch)
    test2()
    test()

    # 测试各个种类的准确率
    def test_kind():

        corr_kind = np.zeros(10, dtype=int)
        sum_kind = np.zeros(10, dtype=int)
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)

            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            tmp = pred.eq(target.data.view_as(pred)).cpu()
            #         print(tmp.shape[0])
            tmp = tmp.numpy()
            tar = target.data.cpu()
            tar = tar.numpy()
            #         print('ok')
            for i in range(tmp.shape[0]):
                corr_kind[tar[i]] += tmp[i][0]
                sum_kind[tar[i]] += 1

        for i in range(10):
            print('class: {} Accuracy: {}/{} ({:.2f}%)'.format(i + 1, corr_kind[i], sum_kind[i],
                                                               100. * corr_kind[i] / sum_kind[i]))

    test_kind()
