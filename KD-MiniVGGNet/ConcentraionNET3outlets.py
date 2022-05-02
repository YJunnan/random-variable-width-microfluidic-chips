# coding: utf-8
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd
# import torchvision
# from sklearn.model_selection import train_test_split#,cross_val_score,KFold
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold
import sklearn.model_selection as ms
import math
from torch.optim import lr_scheduler
begin_time = time.time()
from sklearn import preprocessing


EPOCH =200
#step_size = 100
# train the training data n times, to save time, we just train 1 epoch
train_BATCH_SIZE = 32
test_BATCH_SIZE = 64
LR = 0.001

# 操作间隔
train_interval = 240
test_interval = 40

random_seed = 2 # 随机种子，设置后可以得到稳定的随机数
torch.manual_seed(random_seed)
# 导入数据
#读取数据
data = pd.read_csv('randomall.csv', header=None)
#X = joblib.load('randomall2D7225_shuffle.pkl')
data1 = pd.read_csv('randomall.csv', header=None)
physics = 'concentration'
Y = data.iloc[:, 81:84].to_numpy()  # concentration

X = data1.iloc[:, 0:81].to_numpy() # 读取前面保存的数组
X = X.reshape(10232, 1, 9, 9)
torch.set_printoptions(precision=8)
print(X)
print(Y)
#加载数据
train_X, test_X, train_y, test_y = ms.train_test_split(X, Y, test_size=0.25, random_state=32)
train_X = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_y)
test_X = torch.from_numpy(test_X)
test_y = torch.from_numpy(test_y)

#np.savetxt('test_XC.csv', test_X.reshape(2558,81), delimiter = ',')
#np.savetxt('test_yC.csv', test_y.reshape(2558,3), delimiter = ',')
#np.savetxt('train_Xc.csv', train_X.reshape(7674,81), delimiter = ',')
#np.savetxt('train_yc.csv', train_y.reshape(7674,3), delimiter = ',')
test_X = (test_X).cuda()
test_y = (test_y).cuda()
train_X = (train_X).cuda()
train_y = (train_y).cuda()
print(test_y)
train_torch_dataset = Data.TensorDataset(train_X, train_y)
test_torch_dataset = Data.TensorDataset(test_X, test_y)

#将数据付给batch加载器中
train_loader = Data.DataLoader(
    dataset=train_torch_dataset,      # 数据，封装进Data.TensorDataset()类的数据
    batch_size=train_BATCH_SIZE,      # 每块的大小
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    # num_workers=0,              # 多进程（multiprocess）来读数据
)

test_loader = Data.DataLoader(
    dataset=test_torch_dataset,      # 数据，封装进Data.TensorDataset()类的数据
    batch_size=test_BATCH_SIZE,      # 每块的大小
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    # num_workers=0,              # 多进程（multiprocess）来读数据
)
#定义网络结构
class GooGleNet(nn.Module):
    def __init__(self, num_classes=3, init_weights=False):
        super(GooGleNet, self).__init__()
        self.features = nn.Sequential(  #打包
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            #(32,9,9)
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            #(32,8,8)
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            #(32,9,9)
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # (32,8,8)
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # (32,9,9)
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # (32,8,8)
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # (32,9,9)
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # (32,8,8)
            nn.Conv2d(32, 32, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # (32,9,9)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (32,4,4)
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # (64,4,4)
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # (64,3,3)
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # (64,4,4)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (64,2,2)
        )
        self.classifier = nn.Sequential(
            # 全链接
            nn.Linear(64 * 2 * 2,128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) # 展平或者view()
        x = self.classifier(x)
        return x
#初始化
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                nn.init.constant_(m.bias, 0)
#实例化网络
cnn = GooGleNet()
cnn = cnn.cuda()
print(cnn)  # net architecture
#优化器
optimizer = torch.optim.Adam(cnn.parameters(),lr = LR)
# optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.5, dampening=0.5, weight_decay=0.01, nesterov=False)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[50,100],gamma=0.2,last_epoch=-1)
loss_func = nn.MSELoss()
cnn = cnn.double()

# 定义数组
loss_his = []
test_accuracy_his = []
train_accuracy_his = []
test_loss_his = []
save_acc = []
save_loss = []
save_acc_train = []
save_loss_train = []

#训练函数
def train(eopchs):
    cnn.train()
    loss_cu = 0
    train_acc_cu = 0
    for steps, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = cnn(batch_x)
        loss = loss_func(batch_y,output)
        loss.backward()
        optimizer.step()
        train_accuracy = (batch_y-output) / output
        train_accuracy = (1 - torch.mean(torch.abs(torch.as_tensor(train_accuracy))))
        loss_cu += loss
        train_acc_cu += train_accuracy
    loss_cu = loss_cu/train_interval
    train_acc_cu = train_acc_cu/train_interval
    loss_cu = loss_cu.cpu()
    train_acc_cu = train_acc_cu.cpu()
    print(loss_cu,train_acc_cu)
    loss_his.append(loss_cu.data.numpy())
    train_accuracy_his.append(train_acc_cu.data.numpy())
#测试函数
def test(eopchs):
    cnn.eval()
    test_loss_cu = 0
    test_acc_cu = 0
    with torch.no_grad():
        for test_batch_x, test_batch_y in test_loader:
            test_prediction = cnn(test_batch_x)  # 传入这一组 batch，进行前向计算
            test_loss = loss_func(test_batch_y,test_prediction)
            test_accuracy = (test_batch_y-test_prediction) /test_prediction
            test_accuracy = (1 - torch.mean(torch.abs(torch.as_tensor(test_accuracy))))
            test_acc_cu += test_accuracy
            test_loss_cu += test_loss
        test_acc_cu = test_acc_cu / test_interval
        test_loss_cu = test_loss_cu / test_interval
        cpu_test_loss = test_loss_cu.cpu()
        test_cpu_accuracy = test_acc_cu.cpu()
        print(cpu_test_loss, test_cpu_accuracy)
        test_accuracy_his.append(test_cpu_accuracy.data.numpy())
        test_loss_his.append(cpu_test_loss.data.numpy())
        if test_accuracy_his[-1] >= max(test_accuracy_his):
            # print('save network')
            torch.save(cnn, ('./' + physics + '/' + str(begin_time) + 'net.pkl'))
            torch.save(cnn.state_dict(), './model02.pth')

#开始训练
for epochs in range(1, EPOCH + 1):
    print('EPOCH:',epochs)
    train(epochs)
    test(epochs)
    # scheduler.step()


# train loss plot
plt.figure()
plt.plot(loss_his)
plt.xlabel('Steps')
plt.ylabel('Train Loss')
plt.savefig('./' + physics + '/' + str(begin_time) + 'train_loss.png')
np.save('./' + physics + '/' + str(begin_time) + 'train_loss.npy', loss_his)

# Test loss plot
plt.figure()
plt.plot(test_loss_his)
plt.xlabel('Steps')
plt.ylabel('Test Loss')
plt.savefig('./' + physics + '/' + str(begin_time) + 'test_loss.png')
np.save('./' + physics + '/' + str(begin_time) + 'test_loss.npy', test_loss_his)

# accuracy plot
plt.figure()
plt.plot(test_accuracy_his)
plt.xlabel('Steps')
plt.ylabel('test Accuracy')
plt.ylim(0,1)
plt.savefig('./' + physics + '/' + str(begin_time) + 'test_accuracy.png')
np.save('./' + physics + '/' + str(begin_time) + 'test_accuracy.npy', test_accuracy_his)

plt.figure()
plt.plot(train_accuracy_his)
plt.xlabel('Steps')
plt.ylabel('train Accuracy')
plt.ylim(0,1)
plt.savefig('./' + physics + '/' + str(begin_time) + 'train_accuracy.png')
np.save('./' + physics + '/' + str(begin_time) + 'train_accuracy.npy', train_accuracy_his)


# JIANGNAN UNIVERSITY
# Yu Junnan
# 2022/3/9 16:00
