import torch

from learning_skeleton.src.data_loader import create_data_loaders
from learning_skeleton.src.model import SimpleCNN


def evaluate_accuracy(model, data_loader, device):
    # 评估阶段：切换到非训练状态，不更新模型参数。
    # 在原始项目里，这一步通常写成 model.eval()。
    model.train(False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    device = torch.device("cpu")
    _, test_loader = create_data_loaders(batch_size=32)
    model = SimpleCNN(num_classes=7).to(device)

    accuracy = evaluate_accuracy(model, test_loader, device)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
