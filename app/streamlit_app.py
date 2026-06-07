import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.load_model import MODEL_PATH, load_trained_model
from src.predict import CLASS_NAMES, create_prediction_transform


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


def predict_uploaded_image(model, image, device):
    """
    对用户上传的图片进行预测。

    输入：
    - model: 已加载好的 SimpleCNN
    - image: 用户上传的 PIL 图片
    - device: cpu 或 cuda

    输出：
    - predicted_class: 预测表情类别
    - confidence: 预测置信度
    """
    transform = create_prediction_transform()

    image_tensor = transform(image)

    # 单张图片 shape: [1, 48, 48]
    # 模型需要 batch 维度，所以变成 [1, 1, 48, 48]
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[predicted_index.item()]

    return predicted_class, confidence.item()


st.title("Facial Expression Recognition Demo")

st.write(
    "Upload a face image and the model will predict one of seven facial expressions."
)

st.write(
    "Classes: angry, disgust, fear, happy, neutral, sad, surprise"
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please run `python -m src.train` first to generate simple_cnn.pth."
    )
    st.stop()

model, device = get_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded image",
        width=250,
    )

    if st.button("Predict emotion"):
        predicted_class, confidence = predict_uploaded_image(
            model=model,
            image=image,
            device=device,
        )

        st.subheader("Prediction Result")
        st.write(f"Predicted emotion: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.4f}**")