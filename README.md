# Vision Transformer (ViT) CIFAR-10 分类项目

本项目实现了基于 Vision Transformer (ViT) 的 CIFAR-10 图像分类任务，适合深度学习入门和 ViT 结构理解。

## 项目结构

```
VIT/
├── model/
│   ├── self_write_vit.py      # ViT模型实现
│   ├── data_process.py        # 数据集加载与预处理
├── config/
│   └── vit_config.py          # 模型与训练参数配置
├── checkpoint/
│   └── vit_cifar10.pth        # 训练好的模型权重（自动保存）
├── train_vit.py               # 训练与测试主程序
├── readme.md                  # 项目说明
```

## 环境依赖

- Python 3.8+
- PyTorch
- torchvision

安装依赖：
```bash
pip install torch torchvision
```

## 数据集

本项目自动下载并使用 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 数据集，无需手动准备。

## 主要功能

- Vision Transformer 模型结构自定义实现
- 支持灵活调整 patch 大小、嵌入维度、层数等参数
- 数据增强（随机裁剪、翻转、归一化）
- 自动保存与加载模型权重
- 训练与测试准确率输出

## 快速开始

1. 运行训练主程序：

```bash
python train_vit.py
```

2. 训练过程中会自动保存模型到 `checkpoint/vit_cifar10.pth`，并输出每轮准确率。

## 推荐参数设置

```python
model = VisionTransformer(
    img_size=32,
    patch_size=4,
    in_chans=3,
    num_classes=10,
    embed_dim=128,
    depth=4,
    num_heads=4,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    drop_path_rate=0.1
)
```

## 训练效果

- 训练 30~50 个 epoch，测试准确率可达 70%~85%。
- 可根据硬件和需求调整参数。

## 参考

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [PyTorch 官方文档](https://pytorch.org/)

---

如有问题或建议，欢迎交流！
