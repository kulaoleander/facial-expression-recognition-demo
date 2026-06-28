from pathlib import Path
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def count_images(class_dir):
    # 统计一个类别文件夹里有多少张图片。
    image_count = 0

    for file_path in class_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_count += 1

    return image_count


def inspect_split(split_name):
    # 检查 train 或 test 文件夹中每个类别的图片数量。
    split_dir = DATA_DIR / split_name

    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split folder: {split_dir}")

    results = {}

    for class_name in EXPECTED_CLASSES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        results[class_name] = count_images(class_dir)

    return results


def find_first_image(split_name):
    # 找到一张样本图片，用来检查图片格式。
    split_dir = DATA_DIR / split_name

    for class_name in EXPECTED_CLASSES:
        class_dir = split_dir / class_name
        for file_path in class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                return file_path

    raise FileNotFoundError(f"No image found in {split_dir}")


def inspect_sample_image(image_path):
    # 检查图片 mode 和 size；这会影响 CNN 输入通道和输入尺寸。
    with Image.open(image_path) as image:
        return {
            "path": str(image_path),
            "mode": image.mode,
            "size": image.size,
        }


def main():
    train_results = inspect_split("train")
    test_results = inspect_split("test")
    sample_image_path = find_first_image("train")
    sample_info = inspect_sample_image(sample_image_path)

    print("Train classes:", train_results)
    print("Test classes:", test_results)
    print("Sample image:", sample_info)


if __name__ == "__main__":
    main()
