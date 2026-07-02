from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def create_face_detector():
    """
    创建 OpenCV Haar Cascade 人脸检测器。

    这个函数在项目主线中的位置：
    - 当前阶段：负责找到图片里的人脸位置
    - 下一阶段：predict.py 会用它先裁剪人脸，再做表情分类

    输出：
    - detector: OpenCV CascadeClassifier 对象
    """
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        raise FileNotFoundError(f"Could not load face cascade: {cascade_path}")

    return detector


def pil_image_to_bgr_array(image):
    """
    把 PIL 图片转换成 OpenCV 使用的 BGR numpy array。

    为什么要做这一步？
    - PIL 常用 RGB
    - OpenCV 常用 BGR
    - 人脸检测函数 detectMultiScale 需要 numpy array

    输入：
    - image: PIL Image

    输出：
    - bgr_image: OpenCV BGR 格式图片
    """
    rgb_image = image.convert("RGB")

    rgb_array = np.array(rgb_image)

    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    return bgr_image


def detect_faces(
    image,
    detector=None,
    scale_factor=1.1,
    min_neighbors=5,
    min_size=(30, 30),
):
    """
    检测图片中的人脸。

    输入：
    - image: PIL Image
    - detector: OpenCV 人脸检测器
    - scale_factor: 图像金字塔缩放比例
    - min_neighbors: 检测框稳定程度
    - min_size: 最小人脸尺寸

    输出：
    - faces: 人脸框列表
      每个 face 是一个字典：
      {
          "x": 左上角 x,
          "y": 左上角 y,
          "width": 宽度,
          "height": 高度
      }
    """
    if detector is None:
        detector = create_face_detector()

    bgr_image = pil_image_to_bgr_array(image)

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    detected_faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    faces = []

    for x, y, width, height in detected_faces:
        faces.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            }
        )

    return faces


def select_largest_face(faces):
    """
    从多个检测到的人脸中选择最大的人脸。

    为什么选最大？
    - 上传图片里可能有多个人
    - 当前表情识别 demo 先处理最主要的人脸
    - 最大脸通常是最接近镜头、最适合预测的脸

    输入：
    - faces: 人脸框列表

    输出：
    - 最大的人脸框
    - 如果没有检测到人脸，返回 None
    """
    if len(faces) == 0:
        return None

    largest_face = max(
        faces,
        key=lambda face: face["width"] * face["height"],
    )

    return largest_face


def crop_face_from_box(image, face_box, padding_ratio=0.2):
    """
    根据人脸框裁剪人脸区域。

    输入：
    - image: PIL Image
    - face_box: 人脸框字典
    - padding_ratio: 给人脸框周围增加一点边距

    输出：
    - cropped_face: 裁剪后的人脸 PIL Image
    """
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be greater than or equal to 0")

    image_width, image_height = image.size

    x = int(face_box["x"])
    y = int(face_box["y"])
    width = int(face_box["width"])
    height = int(face_box["height"])

    padding_x = int(width * padding_ratio)
    padding_y = int(height * padding_ratio)

    left = max(0, x - padding_x)
    top = max(0, y - padding_y)
    right = min(image_width, x + width + padding_x)
    bottom = min(image_height, y + height + padding_y)

    cropped_face = image.crop((left, top, right, bottom))

    return cropped_face


def detect_and_crop_largest_face(
    image,
    detector=None,
    padding_ratio=0.2,
):
    """
    检测图片中的最大人脸，并裁剪出来。

    如果检测到人脸：
    - 返回裁剪后的人脸图

    如果没有检测到人脸：
    - 不报错
    - 返回原图
    - face_found = False

    这样做的原因：
    - 真实用户上传图片时，不一定每次都能检测到脸
    - demo 不应该因为没检测到脸就直接崩溃
    """
    faces = detect_faces(
        image=image,
        detector=detector,
    )

    largest_face = select_largest_face(faces)

    if largest_face is None:
        return {
            "face_found": False,
            "face_image": image,
            "face_box": None,
            "num_faces": 0,
        }

    face_image = crop_face_from_box(
        image=image,
        face_box=largest_face,
        padding_ratio=padding_ratio,
    )

    return {
        "face_found": True,
        "face_image": face_image,
        "face_box": largest_face,
        "num_faces": len(faces),
    }