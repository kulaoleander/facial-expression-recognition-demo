import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data_loader import create_train_val_test_loaders
from src.evaluate import evaluate_accuracy
from src.model import SimpleCNN


# 项目根目录：facial-expression-recognition-demo/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 模型权重输出目录和路径
MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_OUTPUT_PATH = MODEL_OUTPUT_DIR / "simple_cnn.pth"

# 训练曲线图片输出目录和路径
FIGURES_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
TRAINING_CURVES_PATH = FIGURES_OUTPUT_DIR / "training_curves.png"

# 训练历史日志输出目录和路径
LOGS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"
TRAINING_HISTORY_PATH = LOGS_OUTPUT_DIR / "training_history.json"


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


def main():
    """
    多 epoch 训练 + validation 评估 + 保存模型主流程。

    当前目标：
    1. 用 train_loader 训练模型
    2. 训练集使用轻量 data augmentation
    3. 每个 epoch 后用 val_loader 评估 validation accuracy
    4. 记录每个 epoch 的 train loss 和 val accuracy
    5. 保存训练曲线图
    6. 保存训练历史 JSON
    7. 训练结束后保存模型权重

    注意：
    - augmentation 只作用于 train_loader
    - validation loader 不使用 augmentation
    - test_loader 暂时不在训练过程中使用
    """
    device = torch.device("cpu")

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

    model = SimpleCNN(num_classes=7).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    print("Training and validation started")
    print("-" * 40)
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
        model_path=MODEL_OUTPUT_PATH,
    )

    saved_history_path = save_training_history(
        history=history,
        output_path=TRAINING_HISTORY_PATH,
    )

    saved_curves_path = save_training_curves(
        history=history,
        output_path=TRAINING_CURVES_PATH,
    )

    print("-" * 40)
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print(f"Training history saved to: {saved_history_path}")
    print(f"Training curves saved to: {saved_curves_path}")
    print("Training and validation finished")


if __name__ == "__main__":
    main()