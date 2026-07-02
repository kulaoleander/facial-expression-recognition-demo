import pytest
import torch
from PIL import Image

from src.model import SimpleCNN
from src.predict import (
    CLASS_NAMES,
    calculate_prediction_result,
    create_prediction_transform,
    predict_image,
)


def test_create_prediction_transform_returns_correct_shape():
    """
    测试单张图片预测 transform 是否输出正确 shape。

    输入：
    - PIL image

    输出：
    - tensor shape: [1, 48, 48]

    这里的 1 表示灰度图 1 个通道。
    """
    image = Image.new("L", (48, 48))

    transform = create_prediction_transform()
    image_tensor = transform(image)

    assert image_tensor.shape == torch.Size([1, 48, 48])


def test_calculate_prediction_result_returns_top_predictions():
    """
    测试 calculate_prediction_result 是否返回 top-k 预测结果。

    这里直接构造 logits，不依赖真实模型。
    """
    outputs = torch.tensor(
        [
            [0.1, 0.2, 0.3, 3.0, 1.5, 0.4, 2.0],
        ]
    )

    result = calculate_prediction_result(
        outputs=outputs,
        class_names=CLASS_NAMES,
        top_k=3,
        low_confidence_threshold=0.5,
    )

    assert result["predicted_class"] == "happy"
    assert "confidence" in result
    assert "top_predictions" in result
    assert len(result["top_predictions"]) == 3

    assert result["top_predictions"][0]["class_name"] == "happy"
    assert result["top_predictions"][1]["class_name"] == "surprise"
    assert result["top_predictions"][2]["class_name"] == "neutral"


def test_calculate_prediction_result_returns_low_confidence_flag():
    """
    测试最高预测概率低于阈值时，is_low_confidence 是否为 True。
    """
    outputs = torch.tensor(
        [
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        ]
    )

    result = calculate_prediction_result(
        outputs=outputs,
        class_names=CLASS_NAMES,
        top_k=3,
        low_confidence_threshold=0.5,
    )

    assert result["is_low_confidence"] is True


def test_calculate_prediction_result_rejects_invalid_top_k():
    """
    测试 top_k 不合法时是否主动报错。
    """
    outputs = torch.randn(1, 7)

    with pytest.raises(ValueError):
        calculate_prediction_result(
            outputs=outputs,
            class_names=CLASS_NAMES,
            top_k=0,
        )

    with pytest.raises(ValueError):
        calculate_prediction_result(
            outputs=outputs,
            class_names=CLASS_NAMES,
            top_k=10,
        )


def test_predict_image_returns_prediction_result(tmp_path):
    """
    测试 predict_image 是否返回预测结果。

    这里使用临时创建的假图片，
    不依赖真实 FER2013 数据集。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)

    image_path = tmp_path / "test_image.jpg"
    image = Image.new("L", (48, 48))
    image.save(image_path)

    result = predict_image(
        model=model,
        image_path=image_path,
        class_names=CLASS_NAMES,
        device=device,
    )

    assert "image_path" in result
    assert "predicted_class" in result
    assert "confidence" in result
    assert "top_predictions" in result
    assert "is_low_confidence" in result
    assert "low_confidence_threshold" in result

    assert result["predicted_class"] in CLASS_NAMES

    assert result["confidence"] >= 0.0
    assert result["confidence"] <= 1.0

    assert len(result["top_predictions"]) == 3


def test_predict_image_accepts_custom_top_k(tmp_path):
    """
    测试 predict_image 是否支持自定义 top_k。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)

    image_path = tmp_path / "test_image.jpg"
    image = Image.new("L", (48, 48))
    image.save(image_path)

    result = predict_image(
        model=model,
        image_path=image_path,
        class_names=CLASS_NAMES,
        device=device,
        top_k=5,
    )

    assert len(result["top_predictions"]) == 5


def test_predict_image_raises_error_for_missing_image(tmp_path):
    """
    测试图片路径不存在时，predict_image 是否主动抛出 FileNotFoundError。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)

    missing_image_path = tmp_path / "missing_image.jpg"

    with pytest.raises(FileNotFoundError):
        predict_image(
            model=model,
            image_path=missing_image_path,
            class_names=CLASS_NAMES,
            device=device,
        )