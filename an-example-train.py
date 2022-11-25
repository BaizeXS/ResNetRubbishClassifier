# 导入库文件
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *

# 0. 设置参数

# 1. 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=False)

# 2. 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device: {}".format(device))

# 3. 输出数据集信息
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的大小为: {}".format(train_data_size))
print("测试数据集的大小为: {}".format(test_data_size))

# 4. 加载数据至 DataLoader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 5. 设置神经网络
bNet = BriefNet()
bNet.to(device)

# 6. 设置损失函数
loss_func = nn.CrossEntropyLoss()
loss_func.to(device)

# 7. 设置优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(bNet.parameters(), lr=learning_rate)

# 8. Set some parameters for training
# 记录训练的次数
total_train_steps = 0
# 记录测试的次数
total_test_steps = 0
# 设置 epoch
epoch = 10
# 设置 Tensorboard
writer = SummaryWriter("../logs_train")

# 训练
for i in range(epoch):
    print("第 {} 轮训练开始".format(i + 1))

    # Training
    # Set Neural Network to Train
    bNet.train()
    # Start Training
    for data in train_dataloader:
        img, target = data
        img = img.to(device)
        target = target.to(device)
        bNet_output = bNet(img)
        loss = loss_func(bNet_output, target)
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_steps = total_train_steps + 1
        if total_train_steps % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_steps, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_steps)

    # Testing
    # Set Neural Network to Test
    bNet.eval()
    # Start Testing
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, target = data
            img = img.to(device)
            target = target.to(device)
            test_output = bNet(img)
            loss = loss_func(test_output, target)
            total_test_loss = total_test_loss + loss.item()
    print("整体数据集上的Loss：{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_steps)
    total_test_steps = total_test_loss + 1

    # Save the Model
    torch.save(bNet, "../bNetModels/bNet{}.pth".format(i))
    print("模型已保存")

writer.close()
