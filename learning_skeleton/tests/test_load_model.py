import pytest
import torch

from learning_skeleton.src.load_model import load_trained_model
from learning_skeleton.src.model import SimpleCNN


def test_load_trained_model_returns_model(tmp_path):
    # 验证保存的参数可以被加载回同样的模型结构。
    device = torch.device("cpu")
    original_model = SimpleCNN(num_classes=7)

    model_path = tmp_path / "test_model.pth"
    torch.save(original_model.state_dict(), model_path)

    loaded_model = load_trained_model(model_path, device)

    assert isinstance(loaded_model, SimpleCNN)


def test_load_trained_model_requires_file(tmp_path):
    device = torch.device("cpu")
    missing_path = tmp_path / "missing_model.pth"

    with pytest.raises(FileNotFoundError):
        load_trained_model(missing_path, device)
