import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data_loader import create_train_val_test_loaders
from src.evaluate import evaluate_accuracy
from src.model import ImprovedCNN, ResNet18ExpressionModel, SimpleCNN


# 项目根目录：facial-expression-recognition-demo/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 模型权重输出目录
MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"

# 不同模型保存成不同文件，避免覆盖 baseline
MODEL_OUTPUT_FILENAMES = {
    "simple_cnn": "simple_cnn.pth",
    "improved_cnn": "improved_cnn.pth",
    "resnet18": "resnet18.pth",
}

# 训练曲线图片输出目录和路径
FIGURES_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
TRAINING_CURVES_PATH = FIGURES_OUTPUT_DIR / "training_curves.png"

# 训练历史日志输出目录和路径
LOGS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"
TRAINING_HISTORY_PATH = LOGS_OUTPUT_DIR / "training_history.json"

# 实验结果总表路径
EXPERIMENT_RESULTS_PATH = LOGS_OUTPUT_DIR / "experiment_results.csv"


def create_model(model_name, num_classes=7, use_pretrained=False):
    """
    根据 model_name 创建对应的模型。

    支持的模型：
    - simple_cnn
    - improved_cnn
    - resnet18

    use_pretrained:
    - 主要给 ResNet18 使用
    - False：不下载预训练权重，适合测试和离线环境
    - True：使用 ImageNet 预训练权重，适合正式 transfer learning
    """
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_name == "improved_cnn":
        return ImprovedCNN(num_classes=num_classes)

    if model_name == "resnet18":
        return ResNet18ExpressionModel(
            num_classes=num_classes,
            use_pretrained=use_pretrained,
        )

    raise ValueError(
        f"Unknown model_name: {model_name}. "
        f"Expected one of: {list(MODEL_OUTPUT_FILENAMES.keys())}"
    )


def get_model_output_path(model_name):
    """
    根据 model_name 返回对应的模型权重保存路径。
    """
    if model_name not in MODEL_OUTPUT_FILENAMES:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Expected one of: {list(MODEL_OUTPUT_FILENAMES.keys())}"
        )

    filename = MODEL_OUTPUT_FILENAMES[model_name]

    return MODEL_OUTPUT_DIR / filename


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    """
    训练模型一个 epoch。
    """
    model.train()

    total_loss = 0.0
    total_batches = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    average_loss = total_loss / total_batches

    return average_loss


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device,
    num_epochs,
    best_model_path=None,
):
    """
    训练模型多个 epoch，并在每个 epoch 后评估 validation accuracy。

    当前这个函数负责：
    1. 每个 epoch 训练一次模型
    2. 每个 epoch 后用 validation set 评估
    3. 记录 train_loss 和 val_accuracy
    4. 记录 best_val_accuracy 和 best_epoch
    5. 如果 validation accuracy 创新高，就保存当前模型
    """
    history = {
        "train_loss": [],
        "val_accuracy": [],
        "best_val_accuracy": -1.0,
        "best_epoch": 0,
    }

    for epoch in range(num_epochs):
        average_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_accuracy = evaluate_accuracy(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        history["train_loss"].append(average_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > history["best_val_accuracy"]:
            history["best_val_accuracy"] = val_accuracy
            history["best_epoch"] = epoch + 1

            if best_model_path is not None:
                save_model(
                    model=model,
                    model_path=best_model_path,
                )

                print(
                    f"Best model updated at epoch {epoch + 1} | "
                    f"best val accuracy: {val_accuracy:.4f}"
                )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"loss: {average_loss:.4f} | "
            f"val accuracy: {val_accuracy:.4f}"
        )

    return history


def save_model(model, model_path):
    """
    保存模型权重到指定路径。
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)


def save_training_history(history, output_path):
    """
    保存训练历史到 JSON 文件。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    return output_path


def save_training_curves(history, output_path):
    """
    保存训练曲线图。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_epochs = len(history["train_loss"])
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(8, 6))

    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")

    plt.title("Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def create_experiment_summary(
    model_name,
    history,
    num_epochs,
    batch_size,
    learning_rate,
    validation_ratio,
    use_augmentation,
    use_pretrained,
    model_output_path,
):
    """
    根据一次训练结果创建实验摘要。

    这个函数会整理出 experiment_results.csv 里的一行。
    """
    val_accuracy_history = history["val_accuracy"]

    final_val_accuracy = val_accuracy_history[-1]
    best_val_accuracy = max(val_accuracy_history)

    summary = {
        "model_name": model_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_ratio": validation_ratio,
        "use_augmentation": use_augmentation,
        "use_pretrained": use_pretrained,
        "final_val_accuracy": final_val_accuracy,
        "best_val_accuracy": best_val_accuracy,
        "model_path": str(model_output_path),
    }

    return summary


def save_experiment_summary(summary, output_path):
    """
    把一次实验摘要追加保存到 CSV 文件。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    with open(output_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary.keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(summary)

    return output_path


def main():
    """
    多 epoch 训练 + validation 评估 + 保存 best model 主流程。
    """
    device = torch.device("cpu")

    model_name = "resnet18"
    use_pretrained = True

    num_epochs = 3
    batch_size = 32
    learning_rate = 0.001
    validation_ratio = 0.2
    random_seed = 42
    use_augmentation = True

    train_loader, val_loader, _ = create_train_val_test_loaders(
        batch_size=batch_size,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
        use_augmentation=use_augmentation,
    )

    model = create_model(
        model_name=model_name,
        num_classes=7,
        use_pretrained=use_pretrained,
    ).to(device)

    model_output_path = get_model_output_path(model_name=model_name)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    print("Training and validation started")
    print("-" * 40)
    print(f"Model name: {model_name}")
    print(f"Use pretrained: {use_pretrained}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Validation ratio: {validation_ratio}")
    print(f"Use augmentation: {use_augmentation}")
    print(f"Train subset size: {len(train_loader.dataset)}")
    print(f"Validation subset size: {len(val_loader.dataset)}")
    print("-" * 40)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        best_model_path=model_output_path,
    )

    saved_history_path = save_training_history(
        history=history,
        output_path=TRAINING_HISTORY_PATH,
    )

    saved_curves_path = save_training_curves(
        history=history,
        output_path=TRAINING_CURVES_PATH,
    )

    experiment_summary = create_experiment_summary(
        model_name=model_name,
        history=history,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_ratio=validation_ratio,
        use_augmentation=use_augmentation,
        use_pretrained=use_pretrained,
        model_output_path=model_output_path,
    )

    saved_experiment_results_path = save_experiment_summary(
        summary=experiment_summary,
        output_path=EXPERIMENT_RESULTS_PATH,
    )

    print("-" * 40)
    print(f"Best model saved to: {model_output_path}")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best validation accuracy: {history['best_val_accuracy']:.4f}")
    print(f"Training history saved to: {saved_history_path}")
    print(f"Training curves saved to: {saved_curves_path}")
    print(f"Experiment results saved to: {saved_experiment_results_path}")
    print("Training and validation finished")


if __name__ == "__main__":
    main()