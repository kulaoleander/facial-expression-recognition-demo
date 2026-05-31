import torch

from src.data_loader import create_datasets, create_data_loaders


def test_datasets_have_expected_sizes():
    """
    测试 train 和 test 数据集大小是否符合 FER2013 当前数据。
    """
    train_dataset, test_dataset = create_datasets()

    assert len(train_dataset) == 28709
    assert len(test_dataset) == 7178


def test_dataset_has_expected_classes():
    """
    测试 ImageFolder 是否正确识别 7 个表情类别。
    """
    train_dataset, test_dataset = create_datasets()

    expected_classes = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]

    assert train_dataset.classes == expected_classes
    assert test_dataset.classes == expected_classes


def test_train_loader_returns_correct_batch_shape():
    """
    测试 DataLoader 是否能返回正确形状的一个 batch。

    images shape:
    [batch_size, channels, height, width]

    对当前数据集来说应该是：
    [32, 1, 48, 48]
    """
    train_loader, _ = create_data_loaders(batch_size=32)

    images, labels = next(iter(train_loader))

    assert images.shape == torch.Size([32, 1, 48, 48])
    assert labels.shape == torch.Size([32])


def test_train_loader_returns_correct_tensor_types():
    """
    测试图片和标签的数据类型是否适合后续训练。

    images 应该是 float32。
    labels 应该是 int64，因为 CrossEntropyLoss 需要整数类别标签。
    """
    train_loader, _ = create_data_loaders(batch_size=32)

    images, labels = next(iter(train_loader))

    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_image_values_are_normalized_between_0_and_1():
    """
    测试 ToTensor 是否把像素值转换到了 0 到 1 之间。
    """
    train_loader, _ = create_data_loaders(batch_size=32)

    images, _ = next(iter(train_loader))

    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0