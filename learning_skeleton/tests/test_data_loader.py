import torch

from learning_skeleton.src.data_loader import create_datasets, create_data_loaders


def test_dataset_has_seven_classes():
    # 验证 ImageFolder 是否识别出 7 个表情类别。
    train_dataset, test_dataset = create_datasets()

    expected_classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    assert train_dataset.classes == expected_classes
    assert test_dataset.classes == expected_classes


def test_train_loader_batch_shape():
    # 验证 DataLoader 输出是否能直接送进 CNN。
    train_loader, _ = create_data_loaders(batch_size=32)
    images, labels = next(iter(train_loader))

    assert images.shape == torch.Size([32, 1, 48, 48])
    assert labels.shape == torch.Size([32])
