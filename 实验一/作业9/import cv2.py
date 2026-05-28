import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device("cpu")

# ====================== 数据加载 ======================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_full = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_set, val_set = random_split(train_full, [48000, 12000])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ====================== 模型（复用上次 BaseCNN） ======================
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

# ====================== 统一训练函数 ======================
def train(model, opt_name, lr, epochs=5):
    criterion = nn.CrossEntropyLoss()
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "sgd_momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    for ep in range(epochs):
        model.train()
        tl, ta, n = 0,0,0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()
            tl += loss.item()*x.size(0)
            ta += (out.argmax(1)==y).sum().item()
            n += x.size(0)
        train_loss.append(tl/n)
        train_acc.append(ta/n)

        model.eval()
        vl, va, n = 0,0,0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out,y)
                vl += loss.item()*x.size(0)
                va += (out.argmax(1)==y).sum().item()
                n += x.size(0)
        val_loss.append(vl/n)
        val_acc.append(va/n)

        print(f"Epoch {ep+1} | train {ta/n:.4f} | val {va/n:.4f}")

    test_loss, test_acc = 0,0
    n = 0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            test_loss += criterion(out,y).item()*x.size(0)
            test_acc += (out.argmax(1)==y).sum().item()
            n += x.size(0)
    test_loss /= n
    test_acc /= n
    print(f"TEST ACC: {test_acc:.4f}")
    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc

# ====================== 任务1：复用模型训练 ======================
print("\n===== 任务1：基础模型训练 =====")
model1 = BaseCNN().to(device)
tr_loss, tr_acc, val_loss, val_acc, t_loss, t_acc = train(model1, "adam", 0.001)

# ====================== 任务2：优化器对比 ======================
print("\n===== 任务2：优化器对比 =====")
optimizers = ["sgd", "sgd_momentum", "adam"]
lrs = [0.1, 0.1, 0.001]
opt_results = []
for o, lr in zip(optimizers, lrs):
    print(f"\n--- {o} lr={lr} ---")
    m = BaseCNN().to(device)
    res = train(m, o, lr)
    opt_results.append((o, res))

# ====================== 任务3：学习率对比（Adam） ======================
print("\n===== 任务3：学习率对比 =====")
lrs = [0.1, 0.01, 0.001]
lr_results = []
for lr in lrs:
    print(f"\n--- lr={lr} ---")
    m = BaseCNN().to(device)
    res = train(m, "adam", lr)
    lr_results.append((lr, res))

# ====================== 任务4：卷积核可视化 ======================
print("\n===== 任务4：卷积核可视化 =====")
conv1_weight = model1.features[0].weight.data.cpu()
plt.figure(figsize=(10,5))
for i in range(min(16, 8)):
    plt.subplot(1,8,i+1)
    plt.imshow(conv1_weight[i,0], cmap='gray')
    plt.axis('off')
plt.savefig("task4_kernels.png")
plt.close()

# ====================== 任务5：Feature Map 可视化 ======================
print("\n===== 任务5：Feature Map =====")
img, lbl = test_dataset[0]
img = img.unsqueeze(0).to(device)
feat = model1.features[0](img).detach().cpu()
plt.figure(figsize=(12,4))
for i in range(min(16,8)):
    plt.subplot(1,8,i+1)
    plt.imshow(feat[0,i], cmap='gray')
plt.savefig("task5_featuremap.png")
plt.close()

# ====================== 任务6：错误样本分析 ======================
print("\n===== 任务6：错误样本 =====")
errors = []
model1.eval()
with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        pred = model1(x).argmax(1)
        for i in range(len(y)):
            if pred[i] != y[i]:
                errors.append((x[i][0].cpu(), y[i].item(), pred[i].item()))
                if len(errors)>=8: break
        if len(errors)>=8: break

plt.figure(figsize=(12,3))
for i,(img, t,p) in enumerate(errors[:8]):
    plt.subplot(1,8,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"T:{t}\nP:{p}")
    plt.axis('off')
plt.savefig("task6_errors.png")
plt.close()

# ====================== 任务7：混淆矩阵 ======================
print("\n===== 任务7：混淆矩阵 =====")
all_pred = []
all_true = []
model1.eval()
with torch.no_grad():
    for x,y in test_loader:
        x = x.to(device)
        pred = model1(x).argmax(1).cpu()
        all_pred.extend(pred.numpy())
        all_true.extend(y.numpy())

cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Pred")
plt.ylabel("True")
plt.savefig("task7_cm.png")
plt.close()

print("\n✅ 任务1~7 全部完成！已生成：")
print("task4_kernels.png")
print("task5_featuremap.png")
print("task6_errors.png")
print("task7_cm.png")