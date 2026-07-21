from pathlib import Path

import torch
import torch.nn as nn

from learning_skeleton.src.data_loader import create_data_loaders
from learning_skeleton.src.evaluate import evaluate_accuracy
from learning_skeleton.src.model import SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "models" / "simple_cnn.pth"


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    # 训练一个 epoch：完整看一遍训练集。
    model.train()

    total_loss = 0.0
    total_batches = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. forward: model makes predictions
        outputs = model(images)

        # 2. loss: compare predictions with true labels
        loss = loss_fn(outputs, labels)

        # 3. clear old gradients
        optimizer.zero_grad()

        # 4. backward: calculate gradients
        loss.backward()

        # 5. step: update model parameters
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


def train_model(model, train_loader, test_loader, loss_fn, optimizer, device, num_epochs):
    # 多个 epoch 的主控训练流程。
    for epoch in range(num_epochs):
        average_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_accuracy = evaluate_accuracy(model, test_loader, device)

        print(f"Epoch {epoch + 1}/{num_epochs} | loss: {average_loss:.4f} | accuracy: {test_accuracy:.4f}")


def save_model(model, model_path):
    # 保存模型参数，供后面加载和预测使用。
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)


def main():
    # 训练入口：把数据、模型、loss、optimizer、训练循环串起来。
    device = torch.device("cpu")
    num_epochs = 3
    batch_size = 32
    learning_rate = 0.001

    train_loader, test_loader = create_data_loaders(batch_size=batch_size)
    model = SimpleCNN(num_classes=7).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, test_loader, loss_fn, optimizer, device, num_epochs)
    save_model(model, MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
