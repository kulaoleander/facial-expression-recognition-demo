from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import create_data_loaders
from src.load_model import MODEL_PATH, load_trained_model


# 项目根目录：facial-expression-recognition-demo/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 详细评估图片输出目录
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# confusion matrix 图片保存路径
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"


def collect_predictions(model, data_loader, device):
    """
    收集模型在整个测试集上的真实标签和预测标签。

    这个函数在项目主线中的位置：
    - 前面：模型已经训练好，并且可以加载 simple_cnn.pth
    - 当前：用模型对 test_loader 里的图片进行预测
    - 后面：用 y_true 和 y_pred 计算 confusion matrix / F1 等指标

    输入：
    - model: 已经加载好权重的模型
    - data_loader: 通常是 test_loader
    - device: cpu 或 cuda

    输出：
    - y_true: 所有图片的真实类别编号
    - y_pred: 所有图片的预测类别编号
    """
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # outputs 的 shape 是 [batch_size, 7]
            # 每一行代表一张图片对 7 个类别的预测分数
            outputs = model(images)

            # 取分数最高的类别作为预测结果
            predictions = torch.argmax(outputs, dim=1)

            # 转回 CPU，再转成 Python list，方便给 scikit-learn 使用
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return y_true, y_pred


def calculate_per_class_accuracy(confusion_matrix_result, class_names):
    """
    根据 confusion matrix 计算每个类别自己的 accuracy。

    confusion matrix 的每一行代表一个真实类别。
    例如第 0 行代表真实类别 angry 的所有样本。

    对某一类来说：
    - 对角线上的数字 = 这个类别预测正确的数量
    - 这一整行的总和 = 这个类别真实样本总数

    输入：
    - confusion_matrix_result: sklearn 计算出的 confusion matrix
    - class_names: 类别名称列表

    输出：
    - per_class_accuracy: 字典，key 是类别名，value 是该类别 accuracy
    """
    per_class_accuracy = {}

    for class_index, class_name in enumerate(class_names):
        correct_count = confusion_matrix_result[class_index][class_index]
        total_count = confusion_matrix_result[class_index].sum()

        if total_count == 0:
            accuracy = 0.0
        else:
            accuracy = correct_count / total_count

        per_class_accuracy[class_name] = accuracy

    return per_class_accuracy


def create_detailed_metrics(y_true, y_pred, class_names):
    """
    创建详细评估指标。

    这个函数把 y_true 和 y_pred 转成更有解释力的评估结果。

    输入：
    - y_true: 真实标签列表
    - y_pred: 预测标签列表
    - class_names: 类别名称列表

    输出：
    - metrics: 一个字典，里面包含：
      1. confusion_matrix
      2. classification_report
      3. per_class_accuracy
    """
    labels = list(range(len(class_names)))

    confusion_matrix_result = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    per_class_accuracy = calculate_per_class_accuracy(
        confusion_matrix_result=confusion_matrix_result,
        class_names=class_names,
    )

    metrics = {
        "confusion_matrix": confusion_matrix_result,
        "classification_report": report,
        "per_class_accuracy": per_class_accuracy,
    }

    return metrics


def save_confusion_matrix_plot(confusion_matrix_result, class_names, output_path):
    """
    把 confusion matrix 保存成图片。

    输入：
    - confusion_matrix_result: confusion matrix 数组
    - class_names: 类别名称列表
    - output_path: 图片保存路径

    输出：
    - output_path: 最终保存的图片路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix_result)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    tick_positions = range(len(class_names))
    plt.xticks(tick_positions, class_names, rotation=45, ha="right")
    plt.yticks(tick_positions, class_names)

    # 在每个格子里写上数量，方便直接看图
    for row_index in range(len(class_names)):
        for col_index in range(len(class_names)):
            value = confusion_matrix_result[row_index][col_index]
            plt.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def print_per_class_accuracy(per_class_accuracy):
    """
    打印每个类别的 accuracy。

    输入：
    - per_class_accuracy: 每个类别 accuracy 的字典

    输出：
    - 无返回值，只负责打印
    """
    print()
    print("Per-class accuracy")
    print("-" * 40)

    for class_name, accuracy in per_class_accuracy.items():
        print(f"{class_name}: {accuracy:.4f}")


def main():
    """
    第 13 阶段：Detailed Evaluation 主流程。

    当前流程：
    1. 创建 test_loader
    2. 加载训练好的 SimpleCNN 权重
    3. 收集 y_true 和 y_pred
    4. 计算 confusion matrix / classification report / per-class accuracy
    5. 保存 confusion_matrix.png
    6. 在终端打印结果
    """
    device = torch.device("cpu")

    print("Detailed evaluation started")
    print("-" * 40)
    print(f"Device: {device}")
    print(f"Model path: {MODEL_PATH}")

    _, test_loader = create_data_loaders(batch_size=32)

    class_names = test_loader.dataset.classes

    model = load_trained_model(
        model_path=MODEL_PATH,
        device=device,
    )

    y_true, y_pred = collect_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    metrics = create_detailed_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    print()
    print("Classification report")
    print("-" * 40)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    print_per_class_accuracy(metrics["per_class_accuracy"])

    saved_path = save_confusion_matrix_plot(
        confusion_matrix_result=metrics["confusion_matrix"],
        class_names=class_names,
        output_path=CONFUSION_MATRIX_PATH,
    )

    print()
    print("Confusion matrix saved")
    print("-" * 40)
    print(saved_path)

    print()
    print("Detailed evaluation finished")


if __name__ == "__main__":
    main()