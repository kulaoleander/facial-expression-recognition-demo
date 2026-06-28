from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from learning_skeleton.src.load_model import MODEL_PATH, load_trained_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "test" / "happy" / "PrivateTest_647018.jpg"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def create_prediction_transform():
    # 单张图片预测时，也必须变成训练时同样的输入格式。
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ]
    )


def predict_image(model, image_path, class_names, device):
    # 输入图片路径，输出预测类别和置信度。
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    transform = create_prediction_transform()
    image = Image.open(image_path)

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # [1, 48, 48] -> [1, 1, 48, 48]
    image_tensor = image_tensor.to(device)

    model.train(False)

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
    device = torch.device("cpu")
    model = load_trained_model(MODEL_PATH, device)

    result = predict_image(model, SAMPLE_IMAGE_PATH, CLASS_NAMES, device)

    print("Predicted class:", result["predicted_class"])
    print("Confidence:", result["confidence"])


if __name__ == "__main__":
    main()
