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
from src.model import ImprovedCNN, SimpleCNN


# 项目根目录：facial-expression-recognition-demo/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 模型权重输出目录
MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"

# 不同模型保存成不同文件，避免覆盖 baseline
MODEL_OUTPUT_FILENAMES = {
    "simple_cnn": "simple_cnn.pth",
    "improved_cnn": "improved_cnn.pth",
}

# 训练曲线图片输出目录和路径
FIGURES_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
TRAINING_CURVES_PATH = FIGURES_OUTPUT_DIR / "training_curves.png"

# 训练历史日志输出目录和路径
LOGS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"
TRAINING_HISTORY_PATH = LOGS_OUTPUT_DIR / "training_history.json"

# 实验结果总表路径
EXPERIMENT_RESULTS_PATH = LOGS_OUTPUT_DIR / "experiment_results.csv"


def create_model(model_name, num_classes=7):
    """
    根据 model_name 创建对应的模型。

    这个函数在项目主线中的位置：
    - 前面：model.py 里已经定义了 SimpleCNN 和 ImprovedCNN
    - 当前：train.py 需要根据名字选择训练哪个模型
    - 后面：实验对比阶段会用它切换不同模型

    输入：
    - model_name: 模型名字
      - "simple_cnn"
      - "improved_cnn"
    - num_classes: 输出类别数量，当前 FER2013 是 7 类

    输出：
    - 一个 PyTorch 模型对象
    """
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_name == "improved_cnn":
        return ImprovedCNN(num_classes=num_classes)

    raise ValueError(
        f"Unknown model_name: {model_name}. "
        f"Expected one of: {list(MODEL_OUTPUT_FILENAMES.keys())}"
    )


def get_model_output_path(model_name):
    """
    根据 model_name 返回对应的模型权重保存路径。

    为什么需要这个函数？
    - SimpleCNN 是 baseline，应该保存成 simple_cnn.pth
    - ImprovedCNN 是升级模型，应该保存成 improved_cnn.pth
    - 这样后面才能做公平对比，不会互相覆盖

    输入：
    - model_name: 模型名字

    输出：
    - model_path: 模型权重保存路径
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

    一个 epoch 的意思是：
    模型完整看一遍训练集。

    这个函数做的事情：
    1. 从 train_loader 里一批一批取图片和标签
    2. 用 model 做预测
    3. 用 loss_fn 计算预测和真实标签之间的差距
    4. 用 optimizer 更新模型参数
    5. 返回这一轮训练的平均 loss

    输出：
    - average_loss: 当前 epoch 的平均训练损失
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


def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs):
    """
    训练模型多个 epoch，并在每个 epoch 后评估 validation accuracy。

    这个函数是当前阶段的训练主控流程：
    1. 训练一个 epoch
    2. 在 validation set 上评估 accuracy
    3. 打印当前 epoch 的 train loss 和 validation accuracy
    4. 把每个 epoch 的 loss 和 validation accuracy 保存到 history 里

    为什么不用 test_loader？
    - test set 应该留到最后做最终评估
    - 训练过程中应该用 validation set 来观察模型表现

    输出：
    - history: 字典，记录每个 epoch 的 train_loss 和 val_accuracy
    """
    history = {
        "train_loss": [],
        "val_accuracy": [],
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

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"loss: {average_loss:.4f} | "
            f"val accuracy: {val_accuracy:.4f}"
        )

    return history


def save_model(model, model_path):
    """
    保存模型权重到指定路径。

    注意：
    这里保存的是 model.state_dict()，也就是模型参数，
    不是把整个模型对象直接保存下来。

    这样做更常见，也更适合后面加载模型做预测。

    输入：
    - model: 已训练好的模型
    - model_path: 模型权重保存路径
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)


def save_training_history(history, output_path):
    """
    保存训练历史到 JSON 文件。

    这个函数在项目主线中的作用：
    - 前面：train_model 已经记录了每个 epoch 的 loss 和 validation accuracy
    - 当前：把这些记录保存成 training_history.json
    - 后面：可以用于复查训练过程，也可以用于重新画图或写实验总结

    输入：
    - history: 训练历史字典
    - output_path: JSON 文件保存路径

    输出：
    - output_path: 保存后的 JSON 文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    return output_path


def save_training_curves(history, output_path):
    """
    保存训练曲线图。

    当前保存两条曲线：
    1. train loss curve
    2. validation accuracy curve

    这张图的作用：
    - 看 loss 有没有下降
    - 看 validation accuracy 有没有上升
    - 为后面分析 overfitting / underfitting 做准备

    输入：
    - history: 训练历史字典
    - output_path: 图片保存路径

    输出：
    - output_path: 保存后的图片路径
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
    model_output_path,
):
    """
    根据一次训练结果创建实验摘要。

    这个函数在项目主线中的位置：
    - 前面：train_model 得到了 history
    - 当前：从 history 中提取关键指标
    - 后面：save_experiment_summary 会把这一行保存到 CSV

    输入：
    - model_name: 当前训练的模型名
    - history: 训练历史，包含 train_loss 和 val_accuracy
    - num_epochs: 训练轮数
    - batch_size: batch size
    - learning_rate: 学习率
    - validation_ratio: validation split 比例
    - use_augmentation: 是否使用 data augmentation
    - model_output_path: 模型保存路径

    输出：
    - summary: 一个字典，对应 CSV 表格中的一行
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
        "final_val_accuracy": final_val_accuracy,
        "best_val_accuracy": best_val_accuracy,
        "model_path": str(model_output_path),
    }

    return summary


def save_experiment_summary(summary, output_path):
    """
    把一次实验摘要追加保存到 CSV 文件。

    这个函数在项目主线中的作用：
    - 每训练一次模型，就往 experiment_results.csv 里追加一行
    - 后面可以用这个文件比较不同模型表现

    如果 CSV 文件不存在：
    - 先写入表头
    - 再写入当前实验结果

    如果 CSV 文件已经存在：
    - 直接追加当前实验结果

    输入：
    - summary: create_experiment_summary 生成的一行实验记录
    - output_path: CSV 保存路径

    输出：
    - output_path: 保存后的 CSV 文件路径
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
    多 epoch 训练 + validation 评估 + 保存模型主流程。

    当前目标：
    1. 选择要训练的模型
    2. 用 train_loader 训练模型
    3. 训练集使用轻量 data augmentation
    4. 每个 epoch 后用 val_loader 评估 validation accuracy
    5. 记录每个 epoch 的 train loss 和 val accuracy
    6. 保存训练曲线图
    7. 保存训练历史 JSON
    8. 训练结束后保存对应模型权重
    9. 保存实验摘要到 experiment_results.csv

    注意：
    - augmentation 只作用于 train_loader
    - validation loader 不使用 augmentation
    - test_loader 暂时不在训练过程中使用
    """
    device = torch.device("cpu")

    model_name = "improved_cnn"

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
    )

    save_model(
        model=model,
        model_path=model_output_path,
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
        model_output_path=model_output_path,
    )

    saved_experiment_results_path = save_experiment_summary(
        summary=experiment_summary,
        output_path=EXPERIMENT_RESULTS_PATH,
    )

    print("-" * 40)
    print(f"Model saved to: {model_output_path}")
    print(f"Training history saved to: {saved_history_path}")
    print(f"Training curves saved to: {saved_curves_path}")
    print(f"Experiment results saved to: {saved_experiment_results_path}")
    print("Training and validation finished")


if __name__ == "__main__":
    main()