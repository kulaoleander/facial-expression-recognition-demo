from src.inspect_dataset import (
    EXPECTED_CLASSES,
    find_first_image,
    inspect_sample_image,
    inspect_split,
)


def test_train_split_has_expected_classes():
    """
    测试 train 数据集是否包含预期的 7 个表情类别。
    """
    train_results = inspect_split("train")

    assert list(train_results.keys()) == EXPECTED_CLASSES


def test_test_split_has_expected_classes():
    """
    测试 test 数据集是否包含预期的 7 个表情类别。
    """
    test_results = inspect_split("test")

    assert list(test_results.keys()) == EXPECTED_CLASSES


def test_each_train_class_has_images():
    """
    测试 train 里每个表情类别都至少有 1 张图片。
    """
    train_results = inspect_split("train")

    for class_name, image_count in train_results.items():
        assert image_count > 0, f"{class_name} has no training images"


def test_each_test_class_has_images():
    """
    测试 test 里每个表情类别都至少有 1 张图片。
    """
    test_results = inspect_split("test")

    for class_name, image_count in test_results.items():
        assert image_count > 0, f"{class_name} has no test images"


def test_sample_image_is_grayscale_48x48():
    """
    测试样本图片是否可以打开，并确认它是 48x48 的灰度图。

    这个测试会影响后面的 CNN 设计：
    mode == "L" 说明输入通道数应该是 1。
    size == (48, 48) 说明原始图片尺寸是 48x48。
    """
    sample_image_path = find_first_image("train")
    sample_info = inspect_sample_image(sample_image_path)

    assert sample_info["mode"] == "L"
    assert sample_info["size"] == (48, 48)