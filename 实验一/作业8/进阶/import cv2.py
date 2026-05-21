import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设备
device = torch.device("cpu")

# ---------------------- 公共函数：训练+验证 ----------------------
def train_val(model, train_loader, val_loader, epochs, lr, opt_type="adam"):
    criterion = nn.CrossEntropyLoss()
    if opt_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        tr_loss = train_loss / len(train_loader)
        tr_acc = correct / total

        # 验证
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        va_loss = val_loss / len(val_loader)
        va_acc = correct / total

        train_loss_list.append(tr_loss)
        train_acc_list.append(tr_acc)
        val_loss_list.append(va_loss)
        val_acc_list.append(va_acc)

        print(f"Epoch {epoch+1:2d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}")

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list

# ---------------------- 基础模型（2层CNN） ----------------------
class BaseCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ---------------------- 进阶1：改进CNN（3层+dropout+大FC） ----------------------
class AdvancedCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*3*3, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ---------------------- 任务2：MNIST 数据加载 ----------------------
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_full = datasets.MNIST('./data', train=True, download=True, transform=transform_mnist)
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform_mnist)
train_set, val_set = random_split(mnist_full, [48000, 12000])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# 任务2：样本图
samples, labels = next(iter(train_loader))
plt.figure(figsize=(12,2))
for i in range(8):
    plt.subplot(1,8,i+1)
    plt.imshow(samples[i][0], cmap='gray')
    plt.title(f"{labels[i].item()}")
    plt.axis('off')
plt.savefig("task2_mnist_samples.png")
plt.close()

# ---------------------- 基础模型训练（Adam） ----------------------
print("=== 基础模型（Adam, lr=0.001）===")
base_model = BaseCNN().to(device)
tr_loss, tr_acc, va_loss, va_acc = train_val(base_model, train_loader, val_loader, epochs=5, lr=0.001, opt_type="adam")

# 任务7：曲线
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(tr_loss, label='Train Loss')
plt.plot(va_loss, label='Val Loss')
plt.title('Loss Curve (Base Model)')
plt.legend()
plt.subplot(1,2,2)
plt.plot(tr_acc, label='Train Acc')
plt.plot(va_acc, label='Val Acc')
plt.title('Acc Curve (Base Model)')
plt.legend()
plt.tight_layout()
plt.savefig("task7_base_curve.png")
plt.close()

# 任务6：测试+测试图
def test_model(model, loader):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += nn.CrossEntropyLoss()(out, y).item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss/len(loader), correct/total

test_loss_base, test_acc_base = test_model(base_model, test_loader)
print(f"基础模型 Test Acc: {test_acc_base:.4f}")

test_imgs, test_lbls = next(iter(test_loader))
with torch.no_grad():
    test_preds = base_model(test_imgs).argmax(1)
plt.figure(figsize=(12,2))
for i in range(8):
    plt.subplot(1,8,i+1)
    plt.imshow(test_imgs[i][0], cmap='gray')
    plt.title(f"T:{test_lbls[i]}\nP:{test_preds[i]}")
    plt.axis('off')
plt.savefig("task6_mnist_test.png")
plt.close()

# ---------------------- 进阶1：改进CNN ----------------------
print("\n=== 进阶1：改进CNN ===")
adv_model = AdvancedCNN().to(device)
tr_loss_adv, tr_acc_adv, va_loss_adv, va_acc_adv = train_val(adv_model, train_loader, val_loader, epochs=5, lr=0.001, opt_type="adam")
test_loss_adv, test_acc_adv = test_model(adv_model, test_loader)
print(f"改进模型 Test Acc: {test_acc_adv:.4f}")

# ---------------------- 进阶2：SGD对比 ----------------------
print("\n=== 进阶2：SGD优化器 ===")
base_sgd = BaseCNN().to(device)
_, _, _, _ = train_val(base_sgd, train_loader, val_loader, epochs=5, lr=0.01, opt_type="sgd")
test_loss_sgd, test_acc_sgd = test_model(base_sgd, test_loader)
print(f"SGD Test Acc: {test_acc_sgd:.4f}")

# 优化器对比表
print("\n优化器对比：")
print(f"SGD   lr=0.01  Acc={test_acc_sgd:.4f}")
print(f"Adam  lr=0.001 Acc={test_acc_base:.4f}")

# ---------------------- 进阶3：CIFAR-10 ----------------------
print("\n=== 进阶3：CIFAR-10 ===")
transform_cifar = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
cifar_full = datasets.CIFAR10('./data', train=True, download=True, transform=transform_cifar)
cifar_test = datasets.CIFAR10('./data', train=False, download=True, transform=transform_cifar)
cifar_train, cifar_val = random_split(cifar_full, [40000, 10000])
cifar_train_loader = DataLoader(cifar_train, batch_size=64, shuffle=True)
cifar_val_loader = DataLoader(cifar_val, batch_size=64, shuffle=False)
cifar_test_loader = DataLoader(cifar_test, batch_size=64, shuffle=False)

# CIFAR模型（输入3通道）
cifar_model = BaseCNN(in_channels=3).to(device)
_, _, _, _ = train_val(cifar_model, cifar_train_loader, cifar_val_loader, epochs=5, lr=0.001, opt_type="adam")
test_loss_cifar, test_acc_cifar = test_model(cifar_model, cifar_test_loader)
print(f"CIFAR-10 Test Acc: {test_acc_cifar:.4f}")

# 数据集对比表
print("\n数据集对比：")
print(f"MNIST   灰度 10类 Acc={test_acc_base:.4f}")
print(f"CIFAR-10 彩色 10类 Acc={test_acc_cifar:.4f}")

print("\n✅ 全部任务完成！生成图片：")
print("task2_mnist_samples.png")
print("task7_base_curve.png")
print("task6_mnist_test.png")