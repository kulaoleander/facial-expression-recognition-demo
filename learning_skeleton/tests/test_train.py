import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from learning_skeleton.src.model import SimpleCNN
from learning_skeleton.src.train import train_one_epoch, save_model


def create_dummy_loader(batch_size=16):
    # 用假数据测试训练流程，不依赖真实 FER2013 数据集。
    images = torch.randn(64, 1, 48, 48)
    labels = torch.randint(0, 7, (64,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_train_one_epoch_returns_loss():
    device = torch.device("cpu")
    model = SimpleCNN(num_classes=7).to(device)
    loader = create_dummy_loader()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    average_loss = train_one_epoch(model, loader, loss_fn, optimizer, device)

    assert isinstance(average_loss, float)
    assert average_loss > 0


def test_save_model_creates_file(tmp_path):
    model = SimpleCNN(num_classes=7)
    model_path = tmp_path / "model.pth"

    save_model(model, model_path)

    assert model_path.exists()
