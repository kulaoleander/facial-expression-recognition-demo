import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from learning_skeleton.src.load_model import MODEL_PATH, load_trained_model
from learning_skeleton.src.predict import CLASS_NAMES, create_prediction_transform


st.set_page_config(page_title="Facial Expression Recognition Skeleton", layout="centered")


@st.cache_resource
def get_model():
    # Streamlit 会反复重新运行脚本，所以这里缓存模型。
    device = torch.device("cpu")
    model = load_trained_model(MODEL_PATH, device)
    return model, device


def predict_uploaded_image(model, image, device):
    # 用户上传图片 -> transform -> model -> predicted class
    transform = create_prediction_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.train(False)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[predicted_index.item()]
    return predicted_class, confidence.item()


st.title("Facial Expression Recognition Skeleton")
st.write("Upload a face image and predict one of seven emotion classes.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Train the model first.")
    st.stop()

model, device = get_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=250)

    if st.button("Predict emotion"):
        predicted_class, confidence = predict_uploaded_image(model, image, device)
        st.write("Predicted emotion:", predicted_class)
        st.write("Confidence:", confidence)
