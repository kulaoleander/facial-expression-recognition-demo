import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.face_detection import detect_and_crop_largest_face
from src.load_model import MODEL_PATH, load_trained_model
from src.predict import (
    CLASS_NAMES,
    calculate_prediction_result,
    create_prediction_transform,
)


st.set_page_config(
    page_title="Facial Expression Recognition Demo",
    layout="centered",
)


@st.cache_resource
def get_model():
    """
    加载训练好的模型。

    Streamlit 每次按钮点击、上传文件时都会重新运行脚本。
    使用 st.cache_resource 可以避免反复加载模型。
    """
    device = torch.device("cpu")

    model = load_trained_model(
        model_path=MODEL_PATH,
        device=device,
    )

    return model, device


def predict_uploaded_image(
    model,
    image,
    device,
    top_k=3,
    low_confidence_threshold=0.5,
    use_face_crop=True,
    face_padding_ratio=0.2,
):
    """
    对用户上传的图片进行预测。

    这个函数在项目主线中的位置：
    - 前面：face_detection.py 已经能检测并裁剪人脸
    - 当前：Streamlit demo 可以选择先裁脸再预测
    - 后面：真实应用会用这个流程处理用户上传图片

    输入：
    - model: 已加载好的模型
    - image: 用户上传的 PIL 图片
    - device: cpu 或 cuda
    - top_k: 返回前几个预测结果
    - low_confidence_threshold: 低置信度阈值
    - use_face_crop: 是否先检测并裁剪人脸
    - face_padding_ratio: 裁剪人脸时保留多少周围区域

    输出：
    - result: 预测结果字典
    """
    transform = create_prediction_transform()

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

    # 单张图片 shape: [1, 48, 48]
    # 模型需要 batch 维度，所以变成 [1, 1, 48, 48]
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)

        result = calculate_prediction_result(
            outputs=outputs,
            class_names=CLASS_NAMES,
            top_k=top_k,
            low_confidence_threshold=low_confidence_threshold,
        )

    result["use_face_crop"] = use_face_crop
    result["face_found"] = face_crop_result["face_found"]
    result["face_box"] = face_crop_result["face_box"]
    result["num_faces"] = face_crop_result["num_faces"]
    result["face_image"] = face_crop_result["face_image"]

    return result


st.title("Facial Expression Recognition Demo")

st.write(
    "Upload a face image and the model will predict one of seven facial expressions."
)

st.write(
    "Classes: angry, disgust, fear, happy, neutral, sad, surprise"
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please run `python -m src.train` first to generate the model file."
    )
    st.stop()

model, device = get_model()

use_face_crop = st.checkbox(
    "Use face detection / face crop before prediction",
    value=True,
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width=250,
    )

    if st.button("Predict emotion"):
        result = predict_uploaded_image(
            model=model,
            image=image,
            device=device,
            top_k=3,
            low_confidence_threshold=0.5,
            use_face_crop=use_face_crop,
            face_padding_ratio=0.2,
        )

        st.subheader("Face Detection")

        st.write(f"Use face crop: **{result['use_face_crop']}**")
        st.write(f"Face found: **{result['face_found']}**")
        st.write(f"Number of faces: **{result['num_faces']}**")
        st.write(f"Face box: **{result['face_box']}**")

        if result["use_face_crop"] and result["face_found"]:
            st.success("Face detected. The cropped face below was used for prediction.")
        elif result["use_face_crop"] and not result["face_found"]:
            st.warning(
                "No face was detected. The original uploaded image was used for prediction instead."
            )

        st.image(
            result["face_image"],
            caption="Image used for prediction",
            width=180,
        )

        st.subheader("Prediction Result")

        st.write(f"Predicted emotion: **{result['predicted_class']}**")
        st.write(f"Confidence: **{result['confidence']:.4f}**")
        st.write(f"Low confidence: **{result['is_low_confidence']}**")

        if result["is_low_confidence"]:
            st.warning(
                "The model is not very confident about this prediction. "
                "Please check the top-3 results instead of trusting only the first class."
            )

        st.subheader("Top-3 Predictions")

        for prediction in result["top_predictions"]:
            class_name = prediction["class_name"]
            probability = prediction["probability"]

            st.write(f"- **{class_name}**: {probability:.4f}")