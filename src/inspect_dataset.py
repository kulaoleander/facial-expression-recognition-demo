from pathlib import Path
from PIL import Image


# 项目根目录：facial-expression-recognition-demo/
# __file__ 表示当前文件 src/inspect_dataset.py
# parents[1] 表示往上两级，回到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 原始数据集目录：data/raw/
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# 我们期望 FER2013 数据集中包含的 7 个表情类别
EXPECTED_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# 只统计这些常见图片格式，避免把其他无关文件算进去
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def count_images(class_dir):
    """
    统计某一个类别文件夹里有多少张图片。

    例如：
    data/raw/train/happy/
    这个函数会统计 happy 文件夹里面有多少个 .jpg / .jpeg / .png 文件。
    """
    image_count = 0

    for file_path in class_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_count += 1

    return image_count


def inspect_split(split_name):
    """
    检查某一个数据集划分，比如 train 或 test。

    输入：
    split_name = "train" 或 "test"

    输出：
    一个字典，例如：
    {
        "angry": 3995,
        "happy": 7215,
        ...
    }
    """
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
    """
    在 train 或 test 里面找到第一张可用图片。

    这个函数的作用不是为了训练模型，
    而是为了检查图片能不能正常打开，以及图片是什么格式。
    """
    split_dir = DATA_DIR / split_name

    for class_name in EXPECTED_CLASSES:
        class_dir = split_dir / class_name

        for file_path in class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                return file_path

    raise FileNotFoundError(f"No image found in {split_dir}")


def inspect_sample_image(image_path):
    """
    打开一张图片，并返回它的基本信息。

    mode:
    - L 表示灰度图
    - RGB 表示彩色图

    size:
    - 例如 (48, 48)，表示宽 48，高 48
    """
    with Image.open(image_path) as image:
        return {
            "path": str(image_path),
            "mode": image.mode,
            "size": image.size,
        }


def print_split_summary(split_name, results):
    """
    把某个 split 的统计结果打印出来。
    """
    print()
    print(f"{split_name.upper()} SET")
    print("-" * 40)

    total_images = 0

    for class_name, image_count in results.items():
        print(f"{class_name}: {image_count}")
        total_images += image_count

    print(f"total: {total_images}")


def main():
    """
    主流程：
    1. 检查 train 数据
    2. 检查 test 数据
    3. 打开一张样本图片
    4. 打印图片 mode 和 size
    """
    print("Dataset inspection")
    print("-" * 40)

    train_results = inspect_split("train")
    test_results = inspect_split("test")

    print_split_summary("train", train_results)
    print_split_summary("test", test_results)

    sample_image_path = find_first_image("train")
    sample_info = inspect_sample_image(sample_image_path)

    print()
    print("SAMPLE IMAGE")
    print("-" * 40)
    print(f"path: {sample_info['path']}")
    print(f"mode: {sample_info['mode']}")
    print(f"size: {sample_info['size']}")


if __name__ == "__main__":
    main()