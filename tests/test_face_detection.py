import cv2
import numpy as np
from PIL import Image

from src.face_detection import (
    create_face_detector,
    detect_and_crop_largest_face,
    detect_faces,
    pil_image_to_bgr_array,
    select_largest_face,
    crop_face_from_box,
)


def test_create_face_detector_returns_valid_detector():
    """
    测试 OpenCV 人脸检测器是否能正常加载。
    """
    detector = create_face_detector()

    assert isinstance(detector, cv2.CascadeClassifier)
    assert not detector.empty()


def test_pil_image_to_bgr_array_returns_expected_shape():
    """
    测试 PIL 图片是否能转换成 OpenCV BGR numpy array。
    """
    image = Image.new("RGB", (100, 80), color=(255, 0, 0))

    bgr_image = pil_image_to_bgr_array(image)

    assert isinstance(bgr_image, np.ndarray)
    assert bgr_image.shape == (80, 100, 3)


def test_detect_faces_returns_empty_list_for_blank_image():
    """
    测试空白图片不会检测出人脸。

    这里不用真实人脸图片，是为了让测试稳定。
    """
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))

    faces = detect_faces(image)

    assert isinstance(faces, list)
    assert len(faces) == 0


def test_select_largest_face_returns_largest_box():
    """
    测试 select_largest_face 是否会选择面积最大的人脸框。
    """
    faces = [
        {
            "x": 10,
            "y": 10,
            "width": 20,
            "height": 20,
        },
        {
            "x": 30,
            "y": 30,
            "width": 40,
            "height": 30,
        },
        {
            "x": 50,
            "y": 50,
            "width": 10,
            "height": 10,
        },
    ]

    largest_face = select_largest_face(faces)

    assert largest_face["x"] == 30
    assert largest_face["y"] == 30
    assert largest_face["width"] == 40
    assert largest_face["height"] == 30


def test_crop_face_from_box_returns_cropped_image_with_padding():
    """
    测试 crop_face_from_box 是否能根据人脸框裁剪图片，并正确加入 padding。
    """
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))

    face_box = {
        "x": 20,
        "y": 30,
        "width": 40,
        "height": 20,
    }

    cropped_face = crop_face_from_box(
        image=image,
        face_box=face_box,
        padding_ratio=0.25,
    )

    assert cropped_face.size == (60, 30)


def test_detect_and_crop_largest_face_falls_back_to_original_when_no_face():
    """
    测试没有检测到人脸时，函数是否返回原图并标记 face_found=False。
    """
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))

    result = detect_and_crop_largest_face(image)

    assert result["face_found"] is False
    assert result["face_image"].size == image.size
    assert result["face_box"] is None
    assert result["num_faces"] == 0