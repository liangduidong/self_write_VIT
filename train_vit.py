import torch
import torch.nn as nn
import torch.optim as optim
from model.self_write_vit import VisionTransformer
from model.data_process import train_loader, test_loader
import os

def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    torch.save(model.state_dict(), "./checkpoint/vit_cifar10.pth")
    print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")

def test(model, test_loader, device):
    model.load_state_dict(torch.load("./checkpoint/vit_cifar10.pth"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f"Test Acc: {100.*correct/total:.2f}%")

if __name__ == "__main__":
    print("Building model...")
    # CIFAR-10有10类，需修改num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model = VisionTransformer(
        img_size=32,         # CIFAR-10原始图片尺寸
        patch_size=4,        # 每张图片分为8x8=64个patch，信息量充足
        in_chans=3,          # 彩色图片
        num_classes=10,      # CIFAR-10有10类
        embed_dim=128,       # 嵌入维度，适合小数据集和中等算力
        depth=4,             # Transformer Block层数，适合入门实验
        num_heads=4,         # 注意力头数，和embed_dim匹配
        mlp_ratio=4.0,       # MLP扩展比例，默认即可
        qkv_bias=True,       # 默认即可
        drop_rate=0.2,       # Dropout，防止过拟合
        attn_drop_rate=0.1,  # 注意力Dropout
        drop_path_rate=0     # 路径Dropout
    ).to(device)

    if os.path.exists("./checkpoint/vit_cifar10.pth"):
        vit_model.load_state_dict(torch.load("./checkpoint/vit_cifar10.pth"))
        print("Model loaded from checkpoint.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit_model.parameters(), lr=1e-4)
    epochs = 40
    for epoch in range(1, epochs+1):
        train(epoch, vit_model, train_loader, criterion, optimizer, device)
        test(vit_model, test_loader, device)