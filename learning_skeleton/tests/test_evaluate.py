import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from learning_skeleton.src.evaluate import evaluate_accuracy


class FixedPredictionModel(nn.Module):
    # 用固定输出测试 accuracy 计算是否正确。
    def __init__(self, fixed_outputs):
        super().__init__()
        self.fixed_outputs = fixed_outputs

    def forward(self, x):
        batch_size = x.shape[0]
        return self.fixed_outputs[:batch_size]


def test_evaluate_accuracy_calculates_value():
    device = torch.device("cpu")

    images = torch.randn(4, 1, 48, 48)
    labels = torch.tensor([0, 1, 2, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    fixed_outputs = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        ]
    )

    model = FixedPredictionModel(fixed_outputs).to(device)
    accuracy = evaluate_accuracy(model, loader, device)

    assert accuracy == 0.75
