from pathlib import Path

import torch

from src.model import SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "simple_cnn.pth"


def load_trained_model(model_path, device):
    """
    加载已经训练并保存好的 SimpleCNN 模型。

    输入：
    - model_path: 模型权重文件路径，例如 outputs/models/simple_cnn.pth
    - device: cpu 或 cuda

    输出：
    - 已经加载好权重、并切换到 eval 模式的模型
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = SimpleCNN(num_classes=7).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


def main():
    """
    手动检查模型是否可以从 .pth 文件加载，并正常进行前向传播。
    """
    device = torch.device("cpu")

    model = load_trained_model(
        model_path=MODEL_PATH,
        device=device,
    )

    dummy_images = torch.randn(32, 1, 48, 48)

    with torch.no_grad():
        outputs = model(dummy_images)

    print("Loaded model check")
    print("-" * 40)
    print(f"Model path: {MODEL_PATH}")
    print(f"Output shape: {outputs.shape}")


if __name__ == "__main__":
    main()