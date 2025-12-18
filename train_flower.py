import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

# =====================
# 1. 基本配置
# =====================
data_dir = "flower_data"
batch_size = 32
num_epochs = 20
lr = 1e-3
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =====================
# 2. 数据预处理
# =====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# 3. 数据集加载
# =====================
train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "train"),
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "val"),
    transform=val_test_transform
)
test_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "test"),
    transform=val_test_transform
)

# 确保 train/test/val 类别一致
test_dataset.class_to_idx = train_dataset.class_to_idx
val_dataset.class_to_idx = train_dataset.class_to_idx

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Classes:", train_dataset.classes)

# =====================
# 4. 模型（ResNet18）
# =====================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# =====================
# 多卡训练修改
# =====================
device_ids = [0,1,2,3]   # 你服务器的 GPU 编号列表
if torch.cuda.device_count() > 1:
    print("Using", len(device_ids), "GPUs for training")
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()

# =====================
# 5. 损失 & 优化器
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# =====================
# 6. 训练与验证
# =====================
def train_one_epoch(model, loader):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    loss_sum = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_sum / len(loader), correct / total


best_val_acc = 0

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # 保存模型，去掉 DataParallel 包装，保证单卡也能加载
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), "best_flower_model.pth")
        else:
            torch.save(model.state_dict(), "best_flower_model.pth")
        print("✅ Saved best model")

# =====================
# 7. 测试集评估
# =====================
print("\nTesting best model...")

# 加载模型，自动处理 DataParallel 前缀问题
state_dict = torch.load("best_flower_model.pth")
# 如果权重是多卡保存的，去掉 module 前缀
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
model.load_state_dict(new_state_dict)

test_loss, test_acc = evaluate(model, test_loader)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc : {test_acc:.4f}")
