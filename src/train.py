import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data_loader import create_data_loaders
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
        # 把数据放到指定设备上
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播：图片进入模型，得到 7 类输出分数
        outputs = model(images)

        # 计算 loss：比较模型输出和真实标签
        loss = loss_fn(outputs, labels)

        # 清空上一批数据留下的梯度
        optimizer.zero_grad()

        # 反向传播：根据 loss 计算每个参数应该怎么调整
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # 记录当前 batch 的 loss
        total_loss += loss.item()
        total_batches += 1

    average_loss = total_loss / total_batches

    return average_loss


def train_model(model, train_loader, test_loader, loss_fn, optimizer, device, num_epochs):
    """
    训练模型多个 epoch，并在每个 epoch 后评估 test accuracy。

    这个函数是当前阶段的训练主控流程：
    1. 训练一个 epoch
    2. 在 test set 上评估 accuracy
    3. 打印当前 epoch 的 loss 和 accuracy
    4. 把每个 epoch 的 loss 和 accuracy 保存到 history 里

    注意：
    当前阶段仍然使用 test_loader 做每轮评估。
    这是为了延续前面的 baseline pipeline。

    后续第 15 阶段会改成：
    - train set 用来训练
    - validation set 用来训练过程中评估
    - test set 只用于最终评估

    输出：
    - history: 字典，记录每个 epoch 的 train_loss 和 test_accuracy
    """
    history = {
        "train_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(num_epochs):
        average_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_accuracy = evaluate_accuracy(
            model=model,
            data_loader=test_loader,
            device=device,
        )

        history["train_loss"].append(average_loss)
        history["test_accuracy"].append(test_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"loss: {average_loss:.4f} | "
            f"test accuracy: {test_accuracy:.4f}"
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
    - 前面：train_model 已经记录了每个 epoch 的 loss 和 accuracy
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
    2. test accuracy curve

    这张图的作用：
    - 看 loss 有没有下降
    - 看 accuracy 有没有上升
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
    plt.plot(epochs, history["test_accuracy"], marker="o", label="Test Accuracy")

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
    多 epoch 训练 + 评估 + 保存模型主流程。

    当前目标：
    1. 训练多个 epoch
    2. 每个 epoch 后评估 test accuracy
    3. 记录每个 epoch 的 loss 和 accuracy
    4. 保存训练曲线图
    5. 保存训练历史 JSON
    6. 训练结束后保存模型权重
    """
    device = torch.device("cpu")

    num_epochs = 3
    batch_size = 32
    learning_rate = 0.001

    train_loader, test_loader = create_data_loaders(batch_size=batch_size)

    model = SimpleCNN(num_classes=7).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    print("Training and evaluation started")
    print("-" * 40)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 40)

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
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
    print("Training and evaluation finished")


if __name__ == "__main__":
    main()