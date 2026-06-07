import pytest
import torch

from src.load_model import load_trained_model
from src.model import SimpleCNN


def test_load_trained_model_returns_model(tmp_path):
    """
    测试 load_trained_model 是否能加载模型权重并返回 SimpleCNN。

    这里不依赖真实的 outputs/models/simple_cnn.pth。
    测试会在 tmp_path 里临时创建一个模型权重文件。
    """
    device = torch.device("cpu")

    original_model = SimpleCNN(num_classes=7)

    model_path = tmp_path / "test_model.pth"

    torch.save(original_model.state_dict(), model_path)

    loaded_model = load_trained_model(
        model_path=model_path,
        device=device,
    )

    assert isinstance(loaded_model, SimpleCNN)

    dummy_images = torch.randn(4, 1, 48, 48)

    with torch.no_grad():
        outputs = loaded_model(dummy_images)

    assert outputs.shape == torch.Size([4, 7])


def test_load_trained_model_raises_error_for_missing_file(tmp_path):
    """
    测试当模型文件不存在时，load_trained_model 是否抛出 FileNotFoundError。

    这样可以避免后面预测时出现更难理解的错误。
    """
    device = torch.device("cpu")

    missing_model_path = tmp_path / "missing_model.pth"

    with pytest.raises(FileNotFoundError):
        load_trained_model(
            model_path=missing_model_path,
            device=device,
        )