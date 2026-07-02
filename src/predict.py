from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.face_detection import detect_and_crop_largest_face
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


def calculate_prediction_result(
    outputs,
    class_names,
    top_k=3,
    low_confidence_threshold=0.5,
):
    """
    根据模型输出 logits 计算预测结果。

    这个函数在项目主线中的位置：
    - 前面：model(image_tensor) 得到 logits
    - 当前：把 logits 转成 probability、top-3、low confidence
    - 后面：Streamlit demo 会显示这些结果

    输入：
    - outputs: 模型输出 logits，shape: [batch_size, num_classes]
    - class_names: 类别名称列表
    - top_k: 返回前几个最可能类别，默认 3
    - low_confidence_threshold: 低置信度阈值，默认 0.5

    输出：
    - 一个字典，包含：
      - predicted_class
      - confidence
      - top_predictions
      - is_low_confidence
      - low_confidence_threshold
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if top_k > len(class_names):
        raise ValueError("top_k cannot be greater than number of classes")

    probabilities = torch.softmax(outputs, dim=1)

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k,
        dim=1,
    )

    top_predictions = []

    for probability, class_index in zip(top_probabilities[0], top_indices[0]):
        class_name = class_names[class_index.item()]

        top_predictions.append(
            {
                "class_name": class_name,
                "probability": probability.item(),
            }
        )

    predicted_class = top_predictions[0]["class_name"]
    confidence = top_predictions[0]["probability"]

    is_low_confidence = confidence < low_confidence_threshold

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "is_low_confidence": is_low_confidence,
        "low_confidence_threshold": low_confidence_threshold,
    }


def predict_image(
    model,
    image_path,
    class_names,
    device,
    top_k=3,
    low_confidence_threshold=0.5,
    use_face_crop=False,
    face_padding_ratio=0.2,
):
    """
    对单张图片进行表情预测。

    输入：
    - model: 已加载好权重的模型
    - image_path: 要预测的图片路径
    - class_names: 类别名称列表
    - device: cpu 或 cuda
    - top_k: 返回前几个最可能类别
    - low_confidence_threshold: 低置信度阈值
    - use_face_crop: 是否先检测并裁剪人脸
    - face_padding_ratio: 裁剪人脸时保留多少周围区域

    输出：
    - 一个字典，包含：
      - 图片路径
      - 预测类别
      - 置信度
      - top-k 预测
      - 低置信度状态
      - 人脸检测状态
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    transform = create_prediction_transform()

    image = Image.open(image_path)

    face_crop_result = {
        "face_found": False,
        "face_image": image,
        "face_box": None,
        "num_faces": 0,
    }

    prediction_image = image

    if use_face_crop:
        face_crop_result = detect_and_crop_largest_face(
            image=image,
            padding_ratio=face_padding_ratio,
        )

        prediction_image = face_crop_result["face_image"]

    image_tensor = transform(prediction_image)

    # 模型需要 batch 维度。
    # 单张图片原本是 [1, 48, 48]，
    # unsqueeze(0) 后变成 [1, 1, 48, 48]。
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)

        prediction_result = calculate_prediction_result(
            outputs=outputs,
            class_names=class_names,
            top_k=top_k,
            low_confidence_threshold=low_confidence_threshold,
        )

    prediction_result["image_path"] = str(image_path)
    prediction_result["use_face_crop"] = use_face_crop
    prediction_result["face_found"] = face_crop_result["face_found"]
    prediction_result["face_box"] = face_crop_result["face_box"]
    prediction_result["num_faces"] = face_crop_result["num_faces"]

    return prediction_result


def main():
    """
    手动运行一次单张图片预测。

    注意：
    这里默认预测 test/happy 下面的一张图片。
    预测结果不一定总是 happy，因为当前模型还不一定足够强。
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
        top_k=3,
        low_confidence_threshold=0.5,
        use_face_crop=True,
        face_padding_ratio=0.2,
    )

    print("Single image prediction")
    print("-" * 40)
    print(f"Image path: {result['image_path']}")
    print(f"Use face crop: {result['use_face_crop']}")
    print(f"Face found: {result['face_found']}")
    print(f"Number of faces: {result['num_faces']}")
    print(f"Face box: {result['face_box']}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Low confidence: {result['is_low_confidence']}")
    print("Top predictions:")

    for prediction in result["top_predictions"]:
        class_name = prediction["class_name"]
        probability = prediction["probability"]

        print(f"- {class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()