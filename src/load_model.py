from pathlib import Path

import torch

from src.model import ImprovedCNN, ResNet18ExpressionModel, SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"

MODEL_OUTPUT_FILENAMES = {
    "simple_cnn": "simple_cnn.pth",
    "improved_cnn": "improved_cnn.pth",
    "resnet18": "resnet18.pth",
}

DEFAULT_MODEL_NAME = "resnet18"


def get_model_output_path(model_name):
    """
    根据模型名称返回对应的权重文件路径。

    这个函数在项目主线中的位置：
    - train.py 会把不同模型保存成不同 .pth 文件
    - load_model.py 需要根据模型名找到正确的 .pth 文件
    - predict.py / Streamlit demo 会使用这个路径加载模型
    """
    if model_name not in MODEL_OUTPUT_FILENAMES:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Expected one of: {list(MODEL_OUTPUT_FILENAMES.keys())}"
        )

    filename = MODEL_OUTPUT_FILENAMES[model_name]

    return MODEL_OUTPUT_DIR / filename


MODEL_PATH = get_model_output_path(DEFAULT_MODEL_NAME)


def create_model_for_loading(model_name, num_classes=7):
    """
    根据模型名称创建对应的模型结构。

    注意：
    这里的作用不是训练，而是先创建同样结构的模型，
    然后用 .pth 文件里的 state_dict 覆盖参数。

    对 ResNet18：
    - 这里使用 use_pretrained=False
    - 因为加载本地 resnet18.pth 时不需要重新下载 ImageNet 权重
    """
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_name == "improved_cnn":
        return ImprovedCNN(num_classes=num_classes)

    if model_name == "resnet18":
        return ResNet18ExpressionModel(
            num_classes=num_classes,
            use_pretrained=False,
        )

    raise ValueError(
        f"Unknown model_name: {model_name}. "
        f"Expected one of: {list(MODEL_OUTPUT_FILENAMES.keys())}"
    )


def load_trained_model(
    model_path=None,
    device=None,
    model_name=DEFAULT_MODEL_NAME,
    num_classes=7,
):
    """
    加载已经训练并保存好的模型。

    输入：
    - model_path: 模型权重文件路径。如果不传，就根据 model_name 自动选择。
    - device: cpu 或 cuda。如果不传，默认 cpu。
    - model_name: simple_cnn / improved_cnn / resnet18
    - num_classes: 类别数量，默认 7

    输出：
    - 已经加载好权重、并切换到 eval 模式的模型
    """
    if device is None:
        device = torch.device("cpu")

    if model_path is None:
        model_path = get_model_output_path(model_name)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = create_model_for_loading(
        model_name=model_name,
        num_classes=num_classes,
    ).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


def main():
    """
    手动检查默认模型是否可以从 .pth 文件加载，并正常进行前向传播。
    """
    device = torch.device("cpu")

    model = load_trained_model(
        model_path=MODEL_PATH,
        device=device,
        model_name=DEFAULT_MODEL_NAME,
    )

    dummy_images = torch.randn(32, 1, 48, 48)

    with torch.no_grad():
        outputs = model(dummy_images)

    print("Loaded model check")
    print("-" * 40)
    print(f"Model name: {DEFAULT_MODEL_NAME}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Output shape: {outputs.shape}")


if __name__ == "__main__":
    main()