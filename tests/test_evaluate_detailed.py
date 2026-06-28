import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate_detailed import (
    calculate_per_class_accuracy,
    collect_predictions,
    create_detailed_metrics,
    save_confusion_matrix_plot,
)


CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


class FixedPredictionModel(nn.Module):
    """
    用于测试的假模型。

    它不是真正的 CNN，也不会学习图片特征。
    它只返回我们提前设置好的输出分数。

    这样做的目的：
    - 不依赖真实训练结果
    - 不依赖随机 accuracy
    - 让测试结果稳定可控
    """

    def __init__(self, fixed_outputs):
        super().__init__()
        self.fixed_outputs = fixed_outputs

    def forward(self, x):
        """
        返回固定预测分数。

        输入：
        - x: 假图片 batch

        输出：
        - fixed_outputs 的前 batch_size 行
        """
        batch_size = x.shape[0]
        return self.fixed_outputs[:batch_size]


def test_collect_predictions_returns_true_and_pred_labels():
    """
    测试 collect_predictions 是否能正确收集：

    - y_true：真实标签
    - y_pred：模型预测标签

    这里不用真实 FER2013 数据，而是用假图片和固定输出。
    """
    device = torch.device("cpu")

    images = torch.randn(4, 1, 48, 48)
    labels = torch.tensor([0, 1, 2, 3])

    dataset = TensorDataset(images, labels)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    fixed_outputs = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # prediction: 0
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],  # prediction: 2
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],  # prediction: 2
            [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],  # prediction: 3
        ]
    )

    model = FixedPredictionModel(fixed_outputs=fixed_outputs)

    y_true, y_pred = collect_predictions(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    assert y_true == [0, 1, 2, 3]
    assert y_pred == [0, 2, 2, 3]


def test_calculate_per_class_accuracy_returns_expected_values():
    """
    测试每个类别 accuracy 的计算逻辑。

    confusion matrix 例子：

    第 0 行：
    - 真实 class_0 一共有 3 个
    - 其中 2 个预测正确
    - 所以 class_0 accuracy = 2 / 3

    第 1 行：
    - 真实 class_1 一共有 3 个
    - 其中 3 个预测正确
    - 所以 class_1 accuracy = 1.0
    """
    confusion_matrix_result = np.array(
        [
            [2, 1],
            [0, 3],
        ]
    )

    class_names = ["class_0", "class_1"]

    per_class_accuracy = calculate_per_class_accuracy(
        confusion_matrix_result=confusion_matrix_result,
        class_names=class_names,
    )

    assert per_class_accuracy["class_0"] == pytest.approx(2 / 3)
    assert per_class_accuracy["class_1"] == pytest.approx(1.0)


def test_create_detailed_metrics_returns_expected_keys():
    """
    测试 create_detailed_metrics 是否返回三个核心结果：

    - confusion_matrix
    - classification_report
    - per_class_accuracy
    """
    y_true = [0, 1, 2, 3, 4, 5, 6]
    y_pred = [0, 1, 1, 3, 4, 4, 6]

    metrics = create_detailed_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=CLASS_NAMES,
    )

    assert isinstance(metrics, dict)
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics
    assert "per_class_accuracy" in metrics


def test_create_detailed_metrics_confusion_matrix_shape_is_7_by_7():
    """
    测试 confusion matrix 的 shape 是否是 7x7。

    因为当前项目有 7 个表情类别：
    angry, disgust, fear, happy, neutral, sad, surprise
    """
    y_true = [0, 1, 2, 3, 4, 5, 6]
    y_pred = [0, 1, 1, 3, 4, 4, 6]

    metrics = create_detailed_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=CLASS_NAMES,
    )

    confusion_matrix_result = metrics["confusion_matrix"]

    assert confusion_matrix_result.shape == (7, 7)


def test_create_detailed_metrics_per_class_accuracy_has_7_classes():
    """
    测试 per_class_accuracy 是否包含 7 个类别。

    这里不测试每个类别具体 accuracy 必须是多少，
    因为真实模型结果可能会变化。

    我们只测试输出结构是否正确。
    """
    y_true = [0, 1, 2, 3, 4, 5, 6]
    y_pred = [0, 1, 1, 3, 4, 4, 6]

    metrics = create_detailed_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=CLASS_NAMES,
    )

    per_class_accuracy = metrics["per_class_accuracy"]

    assert len(per_class_accuracy) == 7
    assert set(per_class_accuracy.keys()) == set(CLASS_NAMES)


def test_save_confusion_matrix_plot_creates_png_file(tmp_path):
    """
    测试 save_confusion_matrix_plot 是否能生成 png 文件。

    tmp_path 是 pytest 提供的临时目录。
    用它保存测试图片，不会污染项目里的 outputs 文件夹。
    """
    confusion_matrix_result = np.eye(7, dtype=int)

    output_path = tmp_path / "confusion_matrix.png"

    saved_path = save_confusion_matrix_plot(
        confusion_matrix_result=confusion_matrix_result,
        class_names=CLASS_NAMES,
        output_path=output_path,
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".png"