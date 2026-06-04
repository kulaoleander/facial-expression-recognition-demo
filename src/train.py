import torch
import torch.nn as nn

from src.data_loader import create_data_loaders
from src.evaluate import evaluate_accuracy
from src.model import SimpleCNN


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
    """
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

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"loss: {average_loss:.4f} | "
            f"test accuracy: {test_accuracy:.4f}"
        )


def main():
    """
    多 epoch 训练 + 评估主流程。

    当前目标：
    观察训练多个 epoch 后，loss 和 test accuracy 是否有变化。
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

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
    )

    print("-" * 40)
    print("Training and evaluation finished")


if __name__ == "__main__":
    main()