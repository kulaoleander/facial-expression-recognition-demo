import csv
import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.model import ImprovedCNN, ResNet18ExpressionModel, SimpleCNN
from src.train import (
    calculate_class_weights_from_dataset,
    collect_labels_from_dataset,
    create_experiment_summary,
    create_loss_function,
    create_model,
    create_scheduler,
    create_training_config,
    get_model_output_path,
    parse_training_args,
    save_experiment_summary,
    save_model,
    save_training_config,
    save_training_curves,
    save_training_history,
    select_device,
    should_stop_early,
    step_scheduler,
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


def create_dummy_imbalanced_dataset():
    """
    创建一个简单的不平衡假数据集，用来测试 class weights。
    """
    images = torch.randn(14, 1, 48, 48)

    labels = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            3,
            4,
            5,
        ]
    )

    return TensorDataset(images, labels)


def test_parse_training_args_returns_default_values():
    args = parse_training_args([])

    assert args.model_name == "resnet18"
    assert args.num_epochs == 3
    assert args.batch_size == 32
    assert args.learning_rate == 0.001
    assert args.validation_ratio == 0.2
    assert args.random_seed == 42
    assert args.use_pretrained is True
    assert args.use_augmentation is True
    assert args.use_class_weights is False
    assert args.scheduler_name == "none"
    assert args.scheduler_patience == 3
    assert args.scheduler_factor == 0.5
    assert args.early_stopping_patience == 0
    assert args.device == "auto"


def test_parse_training_args_accepts_custom_values():
    args = parse_training_args(
        [
            "--model",
            "simple_cnn",
            "--epochs",
            "10",
            "--batch-size",
            "8",
            "--lr",
            "0.01",
            "--validation-ratio",
            "0.1",
            "--random-seed",
            "123",
            "--no-pretrained",
            "--no-use-augmentation",
            "--class-weights",
            "--scheduler",
            "reduce_on_plateau",
            "--scheduler-patience",
            "2",
            "--scheduler-factor",
            "0.25",
            "--early-stopping-patience",
            "4",
            "--device",
            "cpu",
        ]
    )

    assert args.model_name == "simple_cnn"
    assert args.num_epochs == 10
    assert args.batch_size == 8
    assert args.learning_rate == 0.01
    assert args.validation_ratio == 0.1
    assert args.random_seed == 123
    assert args.use_pretrained is False
    assert args.use_augmentation is False
    assert args.use_class_weights is True
    assert args.scheduler_name == "reduce_on_plateau"
    assert args.scheduler_patience == 2
    assert args.scheduler_factor == 0.25
    assert args.early_stopping_patience == 4
    assert args.device == "cpu"


def test_select_device_auto_returns_available_device():
    device = select_device("auto")

    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_select_device_cpu_returns_cpu():
    device = select_device("cpu")

    assert device.type == "cpu"


def test_select_device_cuda_raises_error_when_cuda_is_unavailable():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment.")

    with pytest.raises(RuntimeError):
        select_device("cuda")


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


def test_collect_labels_from_dataset_returns_labels_from_tensor_dataset():
    dataset = create_dummy_imbalanced_dataset()

    labels = collect_labels_from_dataset(dataset)

    assert labels == [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5]


def test_collect_labels_from_subset_returns_selected_labels():
    dataset = create_dummy_imbalanced_dataset()
    subset = Subset(dataset, [0, 6, 9, 11])

    labels = collect_labels_from_dataset(subset)

    assert labels == [0, 1, 2, 3]


def test_calculate_class_weights_from_dataset_returns_tensor():
    dataset = create_dummy_imbalanced_dataset()
    device = torch.device("cpu")

    class_weights = calculate_class_weights_from_dataset(
        dataset=dataset,
        num_classes=6,
        device=device,
    )

    assert isinstance(class_weights, torch.Tensor)
    assert class_weights.shape == torch.Size([6])
    assert class_weights.device.type == "cpu"


def test_calculate_class_weights_gives_larger_weight_to_minority_class():
    dataset = create_dummy_imbalanced_dataset()
    device = torch.device("cpu")

    class_weights = calculate_class_weights_from_dataset(
        dataset=dataset,
        num_classes=6,
        device=device,
    )

    majority_class_weight = class_weights[0]
    minority_class_weight = class_weights[5]

    assert minority_class_weight > majority_class_weight


def test_create_loss_function_without_class_weights():
    dataset = create_dummy_imbalanced_dataset()
    device = torch.device("cpu")

    loss_fn, class_weights = create_loss_function(
        use_class_weights=False,
        train_dataset=dataset,
        num_classes=6,
        device=device,
    )

    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    assert class_weights is None
    assert loss_fn.weight is None


def test_create_loss_function_with_class_weights():
    dataset = create_dummy_imbalanced_dataset()
    device = torch.device("cpu")

    loss_fn, class_weights = create_loss_function(
        use_class_weights=True,
        train_dataset=dataset,
        num_classes=6,
        device=device,
    )

    assert isinstance(loss_fn, nn.CrossEntropyLoss)
    assert isinstance(class_weights, torch.Tensor)
    assert loss_fn.weight is not None


def test_create_scheduler_none_returns_none():
    model = SimpleCNN(num_classes=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = create_scheduler(
        scheduler_name="none",
        optimizer=optimizer,
        num_epochs=3,
    )

    assert scheduler is None


def test_create_scheduler_reduce_on_plateau_returns_scheduler():
    model = SimpleCNN(num_classes=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = create_scheduler(
        scheduler_name="reduce_on_plateau",
        optimizer=optimizer,
        num_epochs=3,
        scheduler_patience=1,
        scheduler_factor=0.5,
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_create_scheduler_cosine_returns_scheduler():
    model = SimpleCNN(num_classes=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = create_scheduler(
        scheduler_name="cosine",
        optimizer=optimizer,
        num_epochs=3,
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_step_scheduler_none_does_not_crash():
    step_scheduler(
        scheduler=None,
        scheduler_name="none",
        val_accuracy=0.5,
    )


def test_should_stop_early_returns_false_when_disabled():
    result = should_stop_early(
        epochs_without_improvement=10,
        early_stopping_patience=0,
    )

    assert result is False


def test_should_stop_early_returns_false_before_patience():
    result = should_stop_early(
        epochs_without_improvement=2,
        early_stopping_patience=3,
    )

    assert result is False


def test_should_stop_early_returns_true_when_patience_reached():
    result = should_stop_early(
        epochs_without_improvement=3,
        early_stopping_patience=3,
    )

    assert result is True


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
    assert "learning_rate" in history
    assert "best_val_accuracy" in history
    assert "best_epoch" in history
    assert "stopped_early" in history
    assert "stopped_epoch" in history

    assert len(history["train_loss"]) == num_epochs
    assert len(history["val_accuracy"]) == num_epochs
    assert len(history["learning_rate"]) == num_epochs

    assert isinstance(history["best_val_accuracy"], float)
    assert isinstance(history["best_epoch"], int)
    assert isinstance(history["stopped_early"], bool)
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


def test_train_model_accepts_scheduler():
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    val_loader = create_dummy_train_loader(batch_size=16)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = create_scheduler(
        scheduler_name="cosine",
        optimizer=optimizer,
        num_epochs=2,
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        scheduler=scheduler,
        scheduler_name="cosine",
    )

    assert len(history["learning_rate"]) == 2


def test_train_model_accepts_early_stopping_argument():
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    val_loader = create_dummy_train_loader(batch_size=16)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        early_stopping_patience=1,
    )

    assert "stopped_early" in history
    assert "stopped_epoch" in history


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
        "learning_rate": [0.001, 0.001, 0.0005],
        "best_val_accuracy": 0.50,
        "best_epoch": 3,
        "stopped_early": False,
        "stopped_epoch": None,
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


def test_save_training_config_creates_json_file(tmp_path):
    config = {
        "model_name": "resnet18",
        "num_epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "use_class_weights": True,
        "scheduler": "reduce_on_plateau",
        "early_stopping_patience": 8,
        "actual_device": "cpu",
    }

    output_path = tmp_path / "training_config.json"

    saved_path = save_training_config(
        config=config,
        output_path=output_path,
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".json"

    with open(saved_path, "r", encoding="utf-8") as file:
        loaded_config = json.load(file)

    assert loaded_config == config


def test_create_training_config_contains_expected_values():
    args = parse_training_args(
        [
            "--model",
            "resnet18",
            "--epochs",
            "50",
            "--batch-size",
            "32",
            "--lr",
            "0.001",
            "--class-weights",
            "--scheduler",
            "reduce_on_plateau",
            "--scheduler-patience",
            "2",
            "--scheduler-factor",
            "0.25",
            "--early-stopping-patience",
            "8",
            "--device",
            "cpu",
        ]
    )

    device = torch.device("cpu")
    class_weights = torch.ones(7)

    config = create_training_config(
        args=args,
        device=device,
        effective_use_pretrained=True,
        class_weights=class_weights,
    )

    assert config["model_name"] == "resnet18"
    assert config["num_epochs"] == 50
    assert config["batch_size"] == 32
    assert config["learning_rate"] == 0.001
    assert config["validation_ratio"] == 0.2
    assert config["random_seed"] == 42
    assert config["use_augmentation"] is True
    assert config["use_pretrained"] is True
    assert config["use_class_weights"] is True
    assert config["class_weights"] == [1.0] * 7
    assert config["scheduler"] == "reduce_on_plateau"
    assert config["scheduler_patience"] == 2
    assert config["scheduler_factor"] == 0.25
    assert config["early_stopping_patience"] == 8
    assert config["requested_device"] == "cpu"
    assert config["actual_device"] == "cpu"


def test_save_training_curves_creates_png_file(tmp_path):
    history = {
        "train_loss": [1.5, 1.2, 1.0],
        "val_accuracy": [0.40, 0.45, 0.50],
        "learning_rate": [0.001, 0.001, 0.0005],
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
        "learning_rate": [0.001, 0.001, 0.0005],
        "best_val_accuracy": 0.46,
        "best_epoch": 2,
        "stopped_early": False,
        "stopped_epoch": None,
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
        use_class_weights=True,
        scheduler_name="reduce_on_plateau",
        early_stopping_patience=8,
        model_output_path=model_output_path,
    )

    assert summary["model_name"] == "improved_cnn"
    assert summary["num_epochs"] == 3
    assert summary["actual_epochs"] == 3
    assert summary["batch_size"] == 32
    assert summary["learning_rate"] == 0.001
    assert summary["validation_ratio"] == 0.2
    assert summary["use_augmentation"] is True
    assert summary["use_pretrained"] is False
    assert summary["use_class_weights"] is True
    assert summary["scheduler"] == "reduce_on_plateau"
    assert summary["early_stopping_patience"] == 8
    assert summary["stopped_early"] is False
    assert summary["stopped_epoch"] is None
    assert summary["final_val_accuracy"] == 0.44
    assert summary["best_val_accuracy"] == 0.46
    assert summary["model_path"].endswith("improved_cnn.pth")


def test_save_experiment_summary_creates_csv_file(tmp_path):
    summary = {
        "model_name": "improved_cnn",
        "num_epochs": 3,
        "actual_epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_ratio": 0.2,
        "use_augmentation": True,
        "use_pretrained": False,
        "use_class_weights": True,
        "scheduler": "reduce_on_plateau",
        "early_stopping_patience": 8,
        "stopped_early": False,
        "stopped_epoch": None,
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
    assert rows[0]["use_class_weights"] == "True"


def test_save_experiment_summary_appends_rows(tmp_path):
    output_path = tmp_path / "experiment_results.csv"

    first_summary = {
        "model_name": "simple_cnn",
        "num_epochs": 3,
        "actual_epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_ratio": 0.2,
        "use_augmentation": True,
        "use_pretrained": False,
        "use_class_weights": False,
        "scheduler": "none",
        "early_stopping_patience": 0,
        "stopped_early": False,
        "stopped_epoch": None,
        "final_val_accuracy": 0.43,
        "best_val_accuracy": 0.44,
        "model_path": "outputs/models/simple_cnn.pth",
    }

    second_summary = {
        "model_name": "resnet18",
        "num_epochs": 50,
        "actual_epochs": 35,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_ratio": 0.2,
        "use_augmentation": True,
        "use_pretrained": True,
        "use_class_weights": True,
        "scheduler": "reduce_on_plateau",
        "early_stopping_patience": 8,
        "stopped_early": True,
        "stopped_epoch": 35,
        "final_val_accuracy": 0.55,
        "best_val_accuracy": 0.60,
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
    assert rows[0]["use_class_weights"] == "False"
    assert rows[1]["model_name"] == "resnet18"
    assert rows[1]["use_pretrained"] == "True"
    assert rows[1]["use_class_weights"] == "True"
