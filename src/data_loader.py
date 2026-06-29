from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# 项目根目录：facial-expression-recognition-demo/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 原始数据目录：data/raw/
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# 训练集和测试集目录
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


def create_basic_transforms():
    """
    创建基础图片预处理流程。

    这个 transform 用于：
    - validation set
    - test set
    - 不使用 augmentation 的普通 train set

    当前做的事情：
    1. 确保图片是 1 通道灰度图
    2. 转成 PyTorch tensor

    输出 tensor 形状：
    [1, 48, 48]
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )


def create_transforms():
    """
    兼容旧代码的基础 transform 函数。

    这个函数现在等价于 create_basic_transforms()。
    """
    return create_basic_transforms()


def create_train_transforms(use_augmentation=False):
    """
    创建训练集 transform。

    如果 use_augmentation=False：
    - 只使用基础 transform

    如果 use_augmentation=True：
    - 使用轻量 data augmentation
    - 只应该用于 train set
    - 不应该用于 validation / test set

    当前 augmentation 包括：
    1. 随机左右翻转
    2. 随机轻微旋转
    3. 随机轻微平移
    4. 转成 tensor
    """
    if not use_augmentation:
        return create_basic_transforms()

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
            ),
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
    这个函数保持基础版本。
    它不使用 augmentation。
    """
    transform = create_basic_transforms()

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
    创建旧版 train 和 test 的 DataLoader。

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


def create_train_val_datasets(
    validation_ratio=0.2,
    random_seed=42,
    use_augmentation=False,
):
    """
    创建 train / validation / test 三个 Dataset。

    这个函数在项目主线中的位置：
    - 前面：已经可以读取原始 train_dataset 和 test_dataset
    - 当前：把原始 train_dataset 拆成 train subset 和 validation subset
    - 后面：train.py 会用 train subset 训练，用 validation subset 做每轮评估

    为什么这里不用 random_split？
    - random_split 会让 train 和 validation 共享同一个 dataset
    - 共享同一个 dataset 就会共享同一个 transform
    - 但我们现在需要 train 可以 augmentation，validation 不可以 augmentation

    所以这里做法是：
    1. 用相同数据路径创建两个 train dataset
    2. 一个用于 train，带 train transform
    3. 一个用于 validation，带 basic transform
    4. 用同一组 indices 拆分它们

    输入：
    - validation_ratio: validation 占原始 train_dataset 的比例
    - random_seed: 固定随机拆分结果，方便复现实验
    - use_augmentation: 是否给 train subset 使用 data augmentation

    输出：
    - train_subset: 真正用于训练的数据
    - val_subset: 训练过程中用于评估的数据
    - test_dataset: 最终测试集
    """
    train_transform = create_train_transforms(use_augmentation=use_augmentation)
    basic_transform = create_basic_transforms()

    train_dataset_for_training = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform,
    )

    train_dataset_for_validation = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=basic_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=basic_transform,
    )

    total_train_size = len(train_dataset_for_training)
    val_size = int(total_train_size * validation_ratio)

    generator = torch.Generator().manual_seed(random_seed)
    indices = torch.randperm(total_train_size, generator=generator).tolist()

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(
        train_dataset_for_training,
        train_indices,
    )

    val_subset = Subset(
        train_dataset_for_validation,
        val_indices,
    )

    return train_subset, val_subset, test_dataset


def create_train_val_test_loaders(
    batch_size=32,
    validation_ratio=0.2,
    random_seed=42,
    use_augmentation=False,
):
    """
    创建 train / validation / test 三个 DataLoader。

    三个 loader 的作用：
    - train_loader: 用来训练模型，shuffle=True
    - val_loader: 用来训练过程中评估模型，shuffle=False
    - test_loader: 用来最终评估模型，shuffle=False

    use_augmentation 的规则：
    - True：只影响 train_loader
    - False：train_loader 也只使用基础 transform
    - validation 和 test 永远不使用 augmentation
    """
    train_subset, val_subset, test_dataset = create_train_val_datasets(
        validation_ratio=validation_ratio,
        random_seed=random_seed,
        use_augmentation=use_augmentation,
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
    6. 检查 augmentation 版本的 loader
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
        use_augmentation=True,
    )

    print()
    print("Train / validation / test split check")
    print("-" * 40)
    print("Use augmentation for train loader: True")
    print(f"Train subset size: {len(train_loader.dataset)}")
    print(f"Validation subset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    train_images, train_labels = next(iter(train_loader))

    print()
    print("One augmented training batch")
    print("-" * 40)
    print(f"Images shape: {train_images.shape}")
    print(f"Labels shape: {train_labels.shape}")
    print(f"Images dtype: {train_images.dtype}")
    print(f"Labels dtype: {train_labels.dtype}")
    print(
        f"Pixel value range: "
        f"{train_images.min().item():.4f} to {train_images.max().item():.4f}"
    )


if __name__ == "__main__":
    main()