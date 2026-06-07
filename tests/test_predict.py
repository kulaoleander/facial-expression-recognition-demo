import pytest
import torch
from PIL import Image

from src.model import SimpleCNN
from src.predict import CLASS_NAMES, create_prediction_transform, predict_image


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

    assert result["predicted_class"] in CLASS_NAMES

    assert result["confidence"] >= 0.0
    assert result["confidence"] <= 1.0


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