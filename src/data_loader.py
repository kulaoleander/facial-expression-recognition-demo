from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
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

    输出：
    - train_dataset: 原始训练集
    - test_dataset: 原始测试集

    注意：
    这个函数保持不变。
    它仍然返回原始 train_dataset 和 test_dataset。
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

    注意：
    这是旧版函数。
    为了不破坏已有代码和测试，暂时保留。

    后续更标准的训练流程会使用：
    create_train_val_test_loaders()
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


def create_train_val_datasets(validation_ratio=0.2, random_seed=42):
    """
    创建 train / validation / test 三个 Dataset。

    这个函数在项目主线中的位置：
    - 前面：已经可以读取原始 train_dataset 和 test_dataset
    - 当前：把原始 train_dataset 再拆成 train subset 和 validation subset
    - 后面：train.py 会用 train subset 训练，用 validation subset 做每轮评估

    为什么要从 train_dataset 里拆 validation？
    - 原始数据只有 train 和 test 两部分
    - test set 应该留到最后做最终评估
    - 所以 validation set 应该从 train set 里拆出来

    输入：
    - validation_ratio: validation 占原始 train_dataset 的比例
    - random_seed: 固定随机拆分结果，方便复现实验

    输出：
    - train_subset: 真正用于训练的数据
    - val_subset: 训练过程中用于评估的数据
    - test_dataset: 最终测试集
    """
    train_dataset, test_dataset = create_datasets()

    total_train_size = len(train_dataset)
    val_size = int(total_train_size * validation_ratio)
    train_size = total_train_size - val_size

    generator = torch.Generator().manual_seed(random_seed)

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    return train_subset, val_subset, test_dataset


def create_train_val_test_loaders(
    batch_size=32,
    validation_ratio=0.2,
    random_seed=42,
):
    """
    创建 train / validation / test 三个 DataLoader。

    这个函数是第 15 阶段之后更标准的 DataLoader 入口。

    三个 loader 的作用：
    - train_loader: 用来训练模型，shuffle=True
    - val_loader: 用来训练过程中评估模型，shuffle=False
    - test_loader: 用来最终评估模型，shuffle=False

    为什么 train_loader 要 shuffle=True？
    - 训练时打乱数据顺序，有助于模型学习更稳定

    为什么 val_loader 和 test_loader 要 shuffle=False？
    - 评估阶段不需要打乱
    - 保持顺序更方便复现和分析

    输出：
    - train_loader
    - val_loader
    - test_loader
    """
    train_subset, val_subset, test_dataset = create_train_val_datasets(
        validation_ratio=validation_ratio,
        random_seed=random_seed,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def main():
    """
    主流程：
    1. 创建 Dataset
    2. 打印类别信息
    3. 创建 DataLoader
    4. 取出一个 batch，检查图片 tensor 的形状
    5. 检查 train / validation / test split
    """
    train_dataset, test_dataset = create_datasets()

    print("Dataset loading check")
    print("-" * 40)

    print(f"Original train dataset size: {len(train_dataset)}")
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

    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        batch_size=32,
        validation_ratio=0.2,
        random_seed=42,
    )

    print()
    print("Train / validation / test split check")
    print("-" * 40)
    print(f"Train subset size: {len(train_loader.dataset)}")
    print(f"Validation subset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")


if __name__ == "__main__":
    main()