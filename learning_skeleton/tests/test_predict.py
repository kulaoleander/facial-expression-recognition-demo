import pytest
import torch
from PIL import Image

from learning_skeleton.src.model import SimpleCNN
from learning_skeleton.src.predict import CLASS_NAMES, create_prediction_transform, predict_image


def test_prediction_transform_shape():
    # 单张图片要变成 [1, 48, 48]，之后再加 batch 维度。
    image = Image.new("L", (48, 48))
    transform = create_prediction_transform()

    image_tensor = transform(image)

    assert image_tensor.shape == torch.Size([1, 48, 48])


def test_predict_image_returns_result(tmp_path):
    device = torch.device("cpu")
    model = SimpleCNN(num_classes=7).to(device)

    image_path = tmp_path / "test_image.jpg"
    Image.new("L", (48, 48)).save(image_path)

    result = predict_image(model, image_path, CLASS_NAMES, device)

    assert result["predicted_class"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_image_requires_existing_file(tmp_path):
    device = torch.device("cpu")
    model = SimpleCNN(num_classes=7).to(device)
    missing_path = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError):
        predict_image(model, missing_path, CLASS_NAMES, device)
