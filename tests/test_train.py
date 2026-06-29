import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import SimpleCNN
from src.train import (
    save_model,
    save_training_curves,
    save_training_history,
    train_model,
    train_one_epoch,
)


def create_dummy_train_loader(batch_size=16):
    """
    创建一个很小的假训练集，用来快速测试训练函数。

    为什么不用真实 FER2013 数据？
    - 真实数据太多，测试会变慢
    - 单元测试只需要验证训练逻辑是否正确
    """
    images = torch.randn(64, 1, 48, 48)
    labels = torch.randint(0, 7, (64,))

    dataset = TensorDataset(images, labels)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return loader


def test_train_one_epoch_returns_float_loss():
    """
    测试 train_one_epoch 是否返回一个 float 类型的 average loss。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    average_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    assert isinstance(average_loss, float)


def test_train_one_epoch_returns_positive_loss():
    """
    测试 average loss 是否是正数。

    对 CrossEntropyLoss 来说，正常情况下 loss 应该大于 0。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    average_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    assert average_loss > 0.0


def test_train_one_epoch_updates_model_parameters():
    """
    测试训练一个 epoch 后，模型参数是否发生变化。

    如果参数完全没有变化，说明 backward 或 optimizer.step 可能没有正常工作。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    parameters_before_training = []

    for parameter in model.parameters():
        if parameter.requires_grad:
            copied_parameter = parameter.clone().detach()
            parameters_before_training.append(copied_parameter)

    train_one_epoch(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    parameters_after_training = []

    for parameter in model.parameters():
        if parameter.requires_grad:
            current_parameter = parameter.detach()
            parameters_after_training.append(current_parameter)

    has_parameter_changed = False

    for before, after in zip(parameters_before_training, parameters_after_training):
        if not torch.equal(before, after):
            has_parameter_changed = True
            break

    assert has_parameter_changed


def test_train_model_runs_multiple_epochs():
    """
    测试 train_model 是否能完成多个 epoch 的训练流程。

    这个测试不追求 accuracy 高低，
    只验证完整流程能跑通：
    train_one_epoch -> evaluate_accuracy -> print result
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    test_loader = create_dummy_train_loader(batch_size=16)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
    )


def test_train_model_returns_history():
    """
    测试 train_model 是否返回训练历史。

    当前第 14 阶段新增了 training curves。
    要画曲线，必须先有 history。

    history 里应该包含：
    - train_loss
    - test_accuracy

    并且每个列表长度应该等于 num_epochs。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    test_loader = create_dummy_train_loader(batch_size=16)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 2

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
    )

    assert isinstance(history, dict)
    assert "train_loss" in history
    assert "test_accuracy" in history
    assert len(history["train_loss"]) == num_epochs
    assert len(history["test_accuracy"]) == num_epochs


def test_save_model_creates_model_file(tmp_path):
    """
    测试 save_model 是否能把模型权重保存成 .pth 文件。

    tmp_path 是 pytest 提供的临时文件夹。
    用它测试不会污染真实 outputs/models 文件夹。
    """
    model = SimpleCNN(num_classes=7)

    model_path = tmp_path / "test_model.pth"

    save_model(
        model=model,
        model_path=model_path,
    )

    assert model_path.exists()
    assert model_path.stat().st_size > 0


def test_save_training_history_creates_json_file(tmp_path):
    """
    测试 save_training_history 是否能生成 JSON 文件。

    这里不用真实训练结果，而是手动创建一个假的 history。
    这样测试更快、更稳定。
    """
    history = {
        "train_loss": [1.5, 1.2, 1.0],
        "test_accuracy": [0.40, 0.45, 0.50],
    }

    output_path = tmp_path / "training_history.json"

    saved_path = save_training_history(
        history=history,
        output_path=output_path,
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".json"

    with open(saved_path, "r", encoding="utf-8") as file:
        loaded_history = json.load(file)

    assert loaded_history == history


def test_save_training_curves_creates_png_file(tmp_path):
    """
    测试 save_training_curves 是否能生成 PNG 图片。

    这里同样使用假的 history。
    测试重点不是曲线长得好不好看，
    而是函数能不能正常生成图片文件。
    """
    history = {
        "train_loss": [1.5, 1.2, 1.0],
        "test_accuracy": [0.40, 0.45, 0.50],
    }

    output_path = tmp_path / "training_curves.png"

    saved_path = save_training_curves(
        history=history,
        output_path=output_path,
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".png"
    assert saved_path.stat().st_size > 0