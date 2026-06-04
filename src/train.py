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
        # 把数据放到指定设备上。
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


def main():
    """
    最小训练 + 评估主流程。

    这一步验证完整闭环：
    DataLoader -> Model -> Loss -> Optimizer -> Train -> Evaluate
    """
    device = torch.device("cpu")

    # 创建训练集和测试集 DataLoader
    train_loader, test_loader = create_data_loaders(batch_size=32)

    # 创建 CNN 模型
    model = SimpleCNN(num_classes=7).to(device)

    # 多分类任务使用 CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss()

    # Adam 负责根据 loss 更新模型参数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    print("Training and evaluation started")
    print("-" * 40)
    print(f"Device: {device}")

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

    print(f"Average training loss: {average_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Training and evaluation finished")


if __name__ == "__main__":
    main()