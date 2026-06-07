from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.load_model import MODEL_PATH, load_trained_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "test" / "happy" / "PrivateTest_647018.jpg"

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def create_prediction_transform():
    """
    创建单张图片预测时的预处理流程。

    这个 transform 要和训练时的输入格式保持一致：
    - 灰度图
    - 48x48
    - tensor
    - 像素值范围 0 到 1
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ]
    )


def predict_image(model, image_path, class_names, device):
    """
    对单张图片进行表情预测。

    输入：
    - model: 已加载好权重的模型
    - image_path: 要预测的图片路径
    - class_names: 类别名称列表
    - device: cpu 或 cuda

    输出：
    - 一个字典，包含图片路径、预测类别和置信度
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    transform = create_prediction_transform()

    image = Image.open(image_path)

    image_tensor = transform(image)

    # 模型需要 batch 维度。
    # 单张图片原本是 [1, 48, 48]，
    # unsqueeze(0) 后变成 [1, 1, 48, 48]。
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_index.item()]

    return {
        "image_path": str(image_path),
        "predicted_class": predicted_class,
        "confidence": confidence.item(),
    }


def main():
    """
    手动运行一次单张图片预测。

    注意：
    这里默认预测 test/happy 下面的一张图片。
    预测结果不一定总是 happy，因为当前模型还是简单 baseline。
    """
    device = torch.device("cpu")

    model = load_trained_model(
        model_path=MODEL_PATH,
        device=device,
    )

    result = predict_image(
        model=model,
        image_path=SAMPLE_IMAGE_PATH,
        class_names=CLASS_NAMES,
        device=device,
    )

    print("Single image prediction")
    print("-" * 40)
    print(f"Image path: {result['image_path']}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()