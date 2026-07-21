from learning_skeleton.src.inspect_dataset import EXPECTED_CLASSES, inspect_sample_image, inspect_split, find_first_image


def test_train_split_has_expected_classes():
    # 验证 train 文件夹包含 7 个表情类别。
    train_results = inspect_split("train")
    assert list(train_results.keys()) == EXPECTED_CLASSES


def test_sample_image_is_grayscale_48x48():
    # 验证样本图片格式能匹配 CNN 输入设计。
    sample_image_path = find_first_image("train")
    sample_info = inspect_sample_image(sample_image_path)

    assert sample_info["mode"] == "L"
    assert sample_info["size"] == (48, 48)
