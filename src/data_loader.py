from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 项目根目录：facial-expression-recognition-demo/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 原始数据目录：data/raw/
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# 训练集和测试集目录
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


def create_transforms():
    """
    创建图片预处理流程。

    当前数据集图片是：
    - 灰度图 mode = L
    - 尺寸是 48x48

    这里先只做最基础的转换：
    1. 确保图片是 1 通道灰度图
    2. 转成 PyTorch tensor

    输出 tensor 形状会是：
    [1, 48, 48]
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )


def create_datasets():
    """
    用 ImageFolder 读取 train 和 test 数据集。

    ImageFolder 的规则：
    - 子文件夹名字就是类别标签
    - 子文件夹里的图片就是该类别的数据
    """
    transform = create_transforms()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=transform,
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=transform,
    )

    return train_dataset, test_dataset


def create_data_loaders(batch_size=32):
    """
    创建 train 和 test 的 DataLoader。

    DataLoader 的作用：
    - 从 Dataset 里读取数据
    - 按 batch 打包
    - 训练时可以一批一批送进模型
    """
    train_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def main():
    """
    主流程：
    1. 创建 Dataset
    2. 打印类别信息
    3. 创建 DataLoader
    4. 取出一个 batch，检查图片 tensor 的形状
    """
    train_dataset, test_dataset = create_datasets()

    print("Dataset loading check")
    print("-" * 40)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Class to index: {train_dataset.class_to_idx}")

    train_loader, _ = create_data_loaders(batch_size=32)

    images, labels = next(iter(train_loader))

    print()
    print("One training batch")
    print("-" * 40)
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Pixel value range: {images.min().item():.4f} to {images.max().item():.4f}")

    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)


if __name__ == "__main__":
    main()