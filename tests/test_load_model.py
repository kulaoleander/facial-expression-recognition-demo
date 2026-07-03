import pytest
import torch

from src.load_model import (
    DEFAULT_MODEL_NAME,
    MODEL_PATH,
    create_model_for_loading,
    get_model_output_path,
    load_trained_model,
)
from src.model import ImprovedCNN, ResNet18ExpressionModel, SimpleCNN


def test_default_model_name_is_resnet18():
    """
    测试当前项目默认加载模型是否为 ResNet18。

    这是 v1.1 收尾的关键点：
    - 训练阶段最佳模型是 ResNet18
    - 推理和 demo 默认也应该使用 ResNet18
    """
    assert DEFAULT_MODEL_NAME == "resnet18"
    assert MODEL_PATH.name == "resnet18.pth"


def test_get_model_output_path_returns_expected_paths():
    """
    测试不同模型名称是否会返回对应的 .pth 文件路径。
    """
    assert get_model_output_path("simple_cnn").name == "simple_cnn.pth"
    assert get_model_output_path("improved_cnn").name == "improved_cnn.pth"
    assert get_model_output_path("resnet18").name == "resnet18.pth"


def test_create_model_for_loading_returns_correct_model_types():
    """
    测试 create_model_for_loading 是否能创建正确模型结构。
    """
    simple_model = create_model_for_loading("simple_cnn")
    improved_model = create_model_for_loading("improved_cnn")
    resnet_model = create_model_for_loading("resnet18")

    assert isinstance(simple_model, SimpleCNN)
    assert isinstance(improved_model, ImprovedCNN)
    assert isinstance(resnet_model, ResNet18ExpressionModel)


def test_load_trained_model_loads_simple_cnn_weights(tmp_path):
    """
    测试 load_trained_model 是否能加载 SimpleCNN 权重。
    """
    device = torch.device("cpu")

    original_model = SimpleCNN(num_classes=7)

    model_path = tmp_path / "simple_cnn.pth"

    torch.save(original_model.state_dict(), model_path)

    loaded_model = load_trained_model(
        model_path=model_path,
        device=device,
        model_name="simple_cnn",
    )

    assert isinstance(loaded_model, SimpleCNN)

    dummy_images = torch.randn(4, 1, 48, 48)

    with torch.no_grad():
        outputs = loaded_model(dummy_images)

    assert outputs.shape == torch.Size([4, 7])


def test_load_trained_model_loads_resnet18_weights(tmp_path):
    """
    测试 load_trained_model 是否能加载 ResNet18 权重。

    注意：
    这里 use_pretrained=False，是为了测试时不下载 ImageNet 权重。
    """
    device = torch.device("cpu")

    original_model = ResNet18ExpressionModel(
        num_classes=7,
        use_pretrained=False,
    )

    model_path = tmp_path / "resnet18.pth"

    torch.save(original_model.state_dict(), model_path)

    loaded_model = load_trained_model(
        model_path=model_path,
        device=device,
        model_name="resnet18",
    )

    assert isinstance(loaded_model, ResNet18ExpressionModel)

    dummy_images = torch.randn(4, 1, 48, 48)

    with torch.no_grad():
        outputs = loaded_model(dummy_images)

    assert outputs.shape == torch.Size([4, 7])


def test_load_trained_model_raises_error_for_missing_file(tmp_path):
    """
    测试当模型文件不存在时，load_trained_model 是否抛出 FileNotFoundError。
    """
    device = torch.device("cpu")

    missing_model_path = tmp_path / "missing_model.pth"

    with pytest.raises(FileNotFoundError):
        load_trained_model(
            model_path=missing_model_path,
            device=device,
            model_name="resnet18",
        )


def test_unknown_model_name_raises_value_error():
    """
    测试未知模型名称是否主动报错。
    """
    with pytest.raises(ValueError):
        get_model_output_path("unknown_model")

    with pytest.raises(ValueError):
        create_model_for_loading("unknown_model")