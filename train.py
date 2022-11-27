import torch
import torch.nn as nn

# from google.colab import drive # 如果使用 Colab，则需要使用该库连接云盘并使用云盘中的数据集
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import RubDataset
from model import *

# 0. 设置参数
learning_rate = 1e-4
batch_size = 64
epochs = 20
file_path = ""
train_path = file_path + "/RubbishClassification/train.json"
val_path = file_path + "/RubbishClassification/val.json"
model_weight_path = file_path + "/resnet34-pre.pth"
save_path = file_path + "/ResNetModels"

# 1. 准备数据集
transform4train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform4test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_data = RubDataset(train_path, transform4train)
test_data = RubDataset(val_path, transform4test)

# 2. 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device: {}".format(device))

# 3. 输出数据集信息
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的大小为: {}".format(train_data_size))
print("测试数据集的大小为: {}".format(test_data_size))

# 4. 加载数据至 DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                          shuffle=True, drop_last=False, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                         shuffle=False, drop_last=False, num_workers=4)

# 5. 设置神经网络
net = resnet34()
net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 16)
net.to(device)
print("网络结构:")
print(net.to(device))

# 6. 设置损失函数
loss_func = nn.CrossEntropyLoss()
loss_func.to(device)

# 7. 设置优化器
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# 8. 训练与验证
torch.cuda.empty_cache()
train_loss_all = []
test_loss_all = []
train_acc_all = []
test_acc_all = []

for i in range(epochs):
    print("第 {} 轮训练开始".format(i + 1))
    # Training
    net.train()  # 设置网络为训练模式
    train_loss = 0  # 初始化训练损失
    train_acc = 0.0  # 初始化训练准确率
    train_num = 0  # 初始化训练数
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        img, target = data
        img = img.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        train_output = net(img)
        loss = loss_func(train_output, target)
        train_output = torch.argmax(train_output, 1)
        loss.backward()
        optimizer.step()

        train_loss += abs(loss.item()) * img.size(0)
        accuracy = torch.sum(train_output == target)
        train_acc += accuracy
        train_num += img.size(0)
        print("Train-Loss: {}, Train-Accuracy: {}".format(train_loss / train_num, train_acc / train_num))
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_acc.double().item() / train_num)

    net.eval()
    test_loss = 0
    test_acc = 0.0
    test_num = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            img, target = data
            img.to(device)
            target.to(device)

            test_output = net(img)
            loss = loss_func(test_output, target)
            test_output = torch.argmax(test_output, 1)

            test_loss += abs(loss.item()) * img.size(0)
            accuracy = torch.sum(test_output == target)
            test_acc += accuracy
            test_num += img.size(0)
            print("Test-Loss: {}, Test-Accuracy: {}".format(test_loss / test_num, test_acc / test_num))
            test_loss_all.append(test_loss / test_num)
            test_acc_all.append(test_acc.double().item() / test_num)

    # Save the Model
    torch.save(net, save_path + "/ResNetModels/ResNet{}.pth".format(i))
    print("模型已保存")
# 9. 绘制图像
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_loss_all, "ro-", label="Train Loss")
plt.plot(range(epochs), test_loss_all, "bs-", label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_acc_all, "ro-", label="Train Accuracy")
plt.plot(range(epochs), test_acc_all, "bs-", label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
