import csv
import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import ImprovedCNN, ResNet18ExpressionModel, SimpleCNN
from src.train import (
    create_experiment_summary,
    create_model,
    get_model_output_path,
    save_experiment_summary,
    save_model,
    save_training_curves,
    save_training_history,
    train_model,
    train_one_epoch,
)


def create_dummy_train_loader(batch_size=16):
    """
    创建一个很小的假训练集，用来快速测试训练函数。
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


def test_create_model_returns_simple_cnn():
    model = create_model(
        model_name="simple_cnn",
        num_classes=7,
    )

    assert isinstance(model, SimpleCNN)


def test_create_model_returns_improved_cnn():
    model = create_model(
        model_name="improved_cnn",
        num_classes=7,
    )

    assert isinstance(model, ImprovedCNN)


def test_create_model_returns_resnet18_expression_model():
    """
    测试 create_model("resnet18") 是否返回 ResNet18ExpressionModel。

    这个测试保护的是：
    train.py 可以通过 model_name 选择 ResNet18 模型。
    """
    model = create_model(
        model_name="resnet18",
        num_classes=7,
        use_pretrained=False,
    )

    assert isinstance(model, ResNet18ExpressionModel)


def test_create_model_raises_error_for_unknown_model_name():
    with pytest.raises(ValueError):
        create_model(
            model_name="unknown_model",
            num_classes=7,
        )


def test_get_model_output_path_returns_expected_filenames():
    simple_model_path = get_model_output_path(model_name="simple_cnn")
    improved_model_path = get_model_output_path(model_name="improved_cnn")
    resnet18_model_path = get_model_output_path(model_name="resnet18")

    assert simple_model_path.name == "simple_cnn.pth"
    assert improved_model_path.name == "improved_cnn.pth"
    assert resnet18_model_path.name == "resnet18.pth"


def test_train_one_epoch_returns_float_loss():
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
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    val_loader = create_dummy_train_loader(batch_size=16)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
    )


def test_train_model_returns_history():
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    val_loader = create_dummy_train_loader(batch_size=16)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 2

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
    )

    assert isinstance(history, dict)
    assert "train_loss" in history
    assert "val_accuracy" in history
    assert "best_val_accuracy" in history
    assert "best_epoch" in history

    assert len(history["train_loss"]) == num_epochs
    assert len(history["val_accuracy"]) == num_epochs

    assert isinstance(history["best_val_accuracy"], float)
    assert isinstance(history["best_epoch"], int)
    assert 1 <= history["best_epoch"] <= num_epochs


def test_train_model_saves_best_model_file(tmp_path):
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    val_loader = create_dummy_train_loader(batch_size=16)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_model_path = tmp_path / "best_model.pth"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        best_model_path=best_model_path,
    )

    assert best_model_path.exists()
    assert best_model_path.stat().st_size > 0

    assert history["best_epoch"] >= 1
    assert history["best_val_accuracy"] >= 0.0


def test_save_model_creates_model_file(tmp_path):
    model = SimpleCNN(num_classes=7)

    model_path = tmp_path / "test_model.pth"

    save_model(
        model=model,
        model_path=model_path,
    )

    assert model_path.exists()
    assert model_path.stat().st_size > 0


def test_save_training_history_creates_json_file(tmp_path):
    history = {
        "train_loss": [1.5, 1.2, 1.0],
        "val_accuracy": [0.40, 0.45, 0.50],
        "best_val_accuracy": 0.50,
        "best_epoch": 3,
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
    history = {
        "train_loss": [1.5, 1.2, 1.0],
        "val_accuracy": [0.40, 0.45, 0.50],
        "best_val_accuracy": 0.50,
        "best_epoch": 3,
    }

    output_path = tmp_path / "training_curves.png"

    saved_path = save_training_curves(
        history=history,
        output_path=output_path,
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".png"
    assert saved_path.stat().st_size > 0


def test_create_experiment_summary_contains_expected_values():
    history = {
        "train_loss": [1.6, 1.4, 1.3],
        "val_accuracy": [0.40, 0.46, 0.44],
        "best_val_accuracy": 0.46,
        "best_epoch": 2,
    }

    model_output_path = get_model_output_path(model_name="improved_cnn")

    summary = create_experiment_summary(
        model_name="improved_cnn",
        history=history,
        num_epochs=3,
        batch_size=32,
        learning_rate=0.001,
        validation_ratio=0.2,
        use_augmentation=True,
        use_pretrained=False,
        model_output_path=model_output_path,
    )

    assert summary["model_name"] == "improved_cnn"
    assert summary["num_epochs"] == 3
    assert summary["batch_size"] == 32
    assert summary["learning_rate"] == 0.001
    assert summary["validation_ratio"] == 0.2
    assert summary["use_augmentation"] is True
    assert summary["use_pretrained"] is False
    assert summary["final_val_accuracy"] == 0.44
    assert summary["best_val_accuracy"] == 0.46
    assert summary["model_path"].endswith("improved_cnn.pth")


def test_save_experiment_summary_creates_csv_file(tmp_path):
    summary = {
        "model_name": "improved_cnn",
        "num_epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_ratio": 0.2,
        "use_augmentation": True,
        "use_pretrained": False,
        "final_val_accuracy": 0.44,
        "best_val_accuracy": 0.46,
        "model_path": "outputs/models/improved_cnn.pth",
    }

    output_path = tmp_path / "experiment_results.csv"

    saved_path = save_experiment_summary(
        summary=summary,
        output_path=output_path,
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".csv"

    with open(saved_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["model_name"] == "improved_cnn"
    assert rows[0]["use_pretrained"] == "False"


def test_save_experiment_summary_appends_rows(tmp_path):
    output_path = tmp_path / "experiment_results.csv"

    first_summary = {
        "model_name": "simple_cnn",
        "num_epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_ratio": 0.2,
        "use_augmentation": True,
        "use_pretrained": False,
        "final_val_accuracy": 0.43,
        "best_val_accuracy": 0.44,
        "model_path": "outputs/models/simple_cnn.pth",
    }

    second_summary = {
        "model_name": "resnet18",
        "num_epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_ratio": 0.2,
        "use_augmentation": True,
        "use_pretrained": True,
        "final_val_accuracy": 0.45,
        "best_val_accuracy": 0.46,
        "model_path": "outputs/models/resnet18.pth",
    }

    save_experiment_summary(
        summary=first_summary,
        output_path=output_path,
    )

    save_experiment_summary(
        summary=second_summary,
        output_path=output_path,
    )

    with open(output_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["model_name"] == "simple_cnn"
    assert rows[0]["use_pretrained"] == "False"
    assert rows[1]["model_name"] == "resnet18"
    assert rows[1]["use_pretrained"] == "True"