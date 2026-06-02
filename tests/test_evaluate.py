import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate_accuracy
from src.model import SimpleCNN


def create_dummy_eval_loader(batch_size=4):
    """
    创建一个很小的假评估数据集。

    这里不用真实 FER2013 数据，因为单元测试要快、稳定、可控。
    """
    images = torch.randn(8, 1, 48, 48)
    labels = torch.randint(0, 7, (8,))

    dataset = TensorDataset(images, labels)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return loader


def test_evaluate_accuracy_returns_float():
    """
    测试 evaluate_accuracy 是否返回 float 类型的 accuracy。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    data_loader = create_dummy_eval_loader(batch_size=4)

    accuracy = evaluate_accuracy(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    assert isinstance(accuracy, float)


def test_evaluate_accuracy_is_between_0_and_1():
    """
    测试 accuracy 是否在 0 到 1 之间。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    data_loader = create_dummy_eval_loader(batch_size=4)

    accuracy = evaluate_accuracy(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    assert accuracy >= 0.0
    assert accuracy <= 1.0


class FixedPredictionModel(nn.Module):
    """
    一个用于测试的假模型。

    它不会真正学习图片特征。
    它只会返回预先设置好的 outputs。

    这样我们可以手动控制预测结果，
    用来验证 evaluate_accuracy 的计算逻辑是否正确。
    """

    def __init__(self, fixed_outputs):
        super().__init__()
        self.fixed_outputs = fixed_outputs

    def forward(self, x):
        """
        返回固定 outputs。

        x 是输入图片 batch。
        这里不真正使用 x，只用它的 batch size 来截取对应数量的 outputs。
        """
        batch_size = x.shape[0]

        return self.fixed_outputs[:batch_size]


def test_evaluate_accuracy_calculates_correct_value():
    """
    测试 evaluate_accuracy 是否能算出正确 accuracy。

    这里我们手动设计：
    labels      = [0, 1, 2, 3]
    predictions = [0, 1, 0, 3]

    预测对了 3 个，总共 4 个。
    所以 accuracy 应该是 0.75。
    """
    device = torch.device("cpu")

    images = torch.randn(4, 1, 48, 48)
    labels = torch.tensor([0, 1, 2, 3])

    dataset = TensorDataset(images, labels)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    fixed_outputs = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # prediction: 0, correct
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # prediction: 1, correct
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # prediction: 0, wrong, label is 2
            [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],  # prediction: 3, correct
        ]
    )

    model = FixedPredictionModel(fixed_outputs=fixed_outputs).to(device)

    accuracy = evaluate_accuracy(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    assert accuracy == 0.75