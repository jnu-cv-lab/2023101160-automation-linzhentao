import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== 任务 2：加载图像数据集 =====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
train_dataset_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 训练集 → 训练集 + 验证集（8:2）
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 显示至少 8 张样本图像 + 标注真实标签
data_iter = iter(train_loader)
images, labels = next(data_iter)

plt.figure(figsize=(12, 2))
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(images[i].squeeze().numpy(), cmap='gray')
    plt.title(f'{labels[i].item()}')
    plt.axis('off')
plt.savefig('task2_samples.png')
plt.close()

# ===================== 任务 3：定义 CNN 模型 =====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

# ===================== 任务 4：训练模型 =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

# 记录曲线
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # ===================== 任务 5：验证模型 =====================
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    # 保存
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1} | "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ===================== 任务 7：绘制训练曲线 =====================
# Loss 曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_loss_list, label='Train Loss', marker='o')
plt.plot(range(1, epochs+1), val_loss_list, label='Val Loss', marker='s')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Acc 曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_acc_list, label='Train Acc', marker='o')
plt.plot(range(1, epochs+1), val_acc_list, label='Val Acc', marker='s')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('task7_curve.png')
plt.close()

# ===================== 任务 6：测试模型 =====================
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

avg_test_loss = test_loss / len(test_loader)
test_acc = test_correct / test_total

print("\n===== 测试集结果 =====")
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 显示 8 张测试图 + 真实标签 + 预测标签
test_images, test_labels = next(iter(test_loader))
with torch.no_grad():
    test_outputs = model(test_images)
    _, test_preds = torch.max(test_outputs, 1)

plt.figure(figsize=(12, 2))
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(test_images[i].squeeze().numpy(), cmap='gray')
    plt.title(f'T:{test_labels[i]}\nP:{test_preds[i]}')
    plt.axis('off')
plt.savefig('task6_test_samples.png')
plt.close()

print("\n✅ 所有任务完成！已生成图片：")
print("1. task2_samples.png")
print("2. task7_curve.png")
print("3. task6_test_samples.png")