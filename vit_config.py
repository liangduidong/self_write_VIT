# Vision Transformer 配置参数
vit_config = {
    "img_size": 224,
    "patch_size": 16,
    "in_chans": 3,
    "num_classes": 10,      # CIFAR-10为10类，CIFAR-100为100类
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.0
}

# 训练相关参数
train_config = {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 3e-4,
    "num_workers": 1,
    "device": "cuda"  # 或 "cpu"
}