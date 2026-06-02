import torch

from src.data_loader import create_data_loaders
from src.model import SimpleCNN


def evaluate_accuracy(model, data_loader, device):
    """
    评估模型在某个数据集上的 accuracy。

    这个函数只做评估，不训练模型。

    输入：
    - model: 需要评估的模型
    - data_loader: 测试集或验证集的 DataLoader
    - device: cpu 或 cuda

    输出：
    - accuracy: 预测正确数量 / 总图片数量
    """
    # 切换到评估模式。
    # 训练时用 model.train()
    # 评估时用 model.eval()
    model.eval()

    correct_predictions = 0
    total_samples = 0

    # 评估阶段不需要计算梯度。
    # 这样更快，也更省内存。
    with torch.no_grad():
        for images, labels in data_loader:
            # 把图片和标签放到指定设备。
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播：得到每张图片的 7 类分数。
            outputs = model(images)

            # 找到每张图片分数最高的类别，作为预测结果。
            predictions = torch.argmax(outputs, dim=1)

            # 统计当前 batch 里预测正确的数量。
            correct_predictions += (predictions == labels).sum().item()

            # 统计当前 batch 的图片总数。
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples

    return accuracy


def main():
    """
    最小评估主流程。

    注意：
    这里的模型是刚创建的随机初始化模型，没有加载训练好的权重。
    所以 accuracy 可能很低，这是正常的。

    当前目标只是验证评估流程能跑通。
    """
    device = torch.device("cpu")

    _, test_loader = create_data_loaders(batch_size=32)

    model = SimpleCNN(num_classes=7).to(device)

    accuracy = evaluate_accuracy(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    print("Evaluation started")
    print("-" * 40)
    print(f"Device: {device}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Evaluation finished")


if __name__ == "__main__":
    main()