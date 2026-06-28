from pathlib import Path

import torch

from learning_skeleton.src.model import SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "simple_cnn.pth"


def load_trained_model(model_path, device):
    # 骨架作用：创建同样的模型结构，然后把已保存的参数放回模型。
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = SimpleCNN(num_classes=7).to(device)

    # 这里在完整版中会读取 .pth 权重文件，并交给 model.load_state_dict(...)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 预测/评估前切换到非训练状态。
    model.train(False)

    return model


def main():
    device = torch.device("cpu")
    model = load_trained_model(MODEL_PATH, device)
    dummy_images = torch.randn(32, 1, 48, 48)

    with torch.no_grad():
        outputs = model(dummy_images)

    print("Output shape:", outputs.shape)


if __name__ == "__main__":
    main()
