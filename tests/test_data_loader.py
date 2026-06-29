import torch

from src.data_loader import (
    create_data_loaders,
    create_datasets,
    create_train_val_datasets,
    create_train_val_test_loaders,
)


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
    测试旧版 train_loader 是否能返回正确形状的一个 batch。

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


def test_train_val_test_datasets_have_expected_sizes():
    """
    测试新的 train / validation / test 数据集大小是否正确。

    原始 train_dataset 有 28709 张图片。
    validation_ratio = 0.2。

    validation size:
    int(28709 * 0.2) = 5741

    train size:
    28709 - 5741 = 22968

    test size:
    原始 test_dataset 保持 7178 不变
    """
    train_subset, val_subset, test_dataset = create_train_val_datasets(
        validation_ratio=0.2,
        random_seed=42,
    )

    assert len(train_subset) == 22968
    assert len(val_subset) == 5741
    assert len(test_dataset) == 7178


def test_train_val_test_loaders_return_correct_batch_shapes():
    """
    测试 train_loader / val_loader / test_loader 是否都能返回正确 batch。

    这一步确认：
    - 新的 train_loader 能训练
    - 新的 val_loader 能评估
    - test_loader 仍然能最终测试
    """
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        batch_size=32,
        validation_ratio=0.2,
        random_seed=42,
    )

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))

    assert train_images.shape == torch.Size([32, 1, 48, 48])
    assert train_labels.shape == torch.Size([32])

    assert val_images.shape == torch.Size([32, 1, 48, 48])
    assert val_labels.shape == torch.Size([32])

    assert test_images.shape == torch.Size([32, 1, 48, 48])
    assert test_labels.shape == torch.Size([32])


def test_train_val_test_loaders_return_valid_tensors():
    """
    测试新的 train / validation / test loader 返回的数据是否适合模型使用。

    检查内容：
    - images 是 float32
    - labels 是 int64
    - 像素值在 0 到 1 之间
    """
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        batch_size=32,
        validation_ratio=0.2,
        random_seed=42,
    )

    loaders = [train_loader, val_loader, test_loader]

    for loader in loaders:
        images, labels = next(iter(loader))

        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64

        assert images.min().item() >= 0.0
        assert images.max().item() <= 1.0