import json
from pathlib import Path

import torch

from src.data_loader import create_train_val_test_loaders
from src.evaluate_detailed import (
    collect_predictions,
    create_detailed_metrics,
    save_confusion_matrix_plot,
)
from src.load_model import DEFAULT_MODEL_NAME, MODEL_PATH, load_trained_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOGS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "logs"
FIGURES_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

FINAL_TEST_METRICS_PATH = LOGS_OUTPUT_DIR / "final_test_metrics.json"
FINAL_TEST_CONFUSION_MATRIX_PATH = FIGURES_OUTPUT_DIR / "final_test_confusion_matrix.png"


def calculate_accuracy_from_predictions(y_true, y_pred):
    """
    根据真实标签和预测标签计算 accuracy。

    这个函数在项目主线中的位置：
    - 前面：模型已经对 test set 做完预测
    - 当前：计算最终 test accuracy
    - 后面：结果会被保存到 final_test_metrics.json 和 README

    输入：
    - y_true: 真实标签列表
    - y_pred: 预测标签列表

    输出：
    - accuracy: 正确预测数量 / 总样本数量
    """
    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    correct_count = 0

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == predicted_label:
            correct_count += 1

    accuracy = correct_count / len(y_true)

    return accuracy


def create_final_test_summary(
    model_name,
    model_path,
    y_true,
    y_pred,
    class_names,
):
    """
    创建 final test evaluation 的结果摘要。

    输入：
    - model_name: 当前评估的模型名称
    - model_path: 当前模型权重路径
    - y_true: test set 真实标签
    - y_pred: test set 预测标签
    - class_names: 类别名称列表

    输出：
    - summary: 可以保存成 JSON 的字典
    """
    test_accuracy = calculate_accuracy_from_predictions(
        y_true=y_true,
        y_pred=y_pred,
    )

    detailed_metrics = create_detailed_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    classification_report = detailed_metrics["classification_report"]
    confusion_matrix = detailed_metrics["confusion_matrix"]
    per_class_accuracy = detailed_metrics["per_class_accuracy"]

    summary = {
        "model_name": model_name,
        "model_path": str(model_path),
        "num_test_samples": len(y_true),
        "test_accuracy": test_accuracy,
        "macro_avg_f1": classification_report["macro avg"]["f1-score"],
        "weighted_avg_f1": classification_report["weighted avg"]["f1-score"],
        "per_class_accuracy": per_class_accuracy,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix.tolist(),
    }

    return summary


def save_final_test_metrics(summary, output_path):
    """
    保存 final test evaluation 结果到 JSON 文件。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    return output_path


def print_final_test_summary(summary):
    """
    在终端打印 final test evaluation 的核心结果。
    """
    print()
    print("Final test summary")
    print("-" * 40)
    print(f"Model name: {summary['model_name']}")
    print(f"Model path: {summary['model_path']}")
    print(f"Number of test samples: {summary['num_test_samples']}")
    print(f"Test accuracy: {summary['test_accuracy']:.4f}")
    print(f"Macro avg F1: {summary['macro_avg_f1']:.4f}")
    print(f"Weighted avg F1: {summary['weighted_avg_f1']:.4f}")

    print()
    print("Per-class accuracy")
    print("-" * 40)

    for class_name, accuracy in summary["per_class_accuracy"].items():
        print(f"{class_name}: {accuracy:.4f}")


def main():
    """
    Final test evaluation 主流程。

    注意：
    这一步不训练模型。
    它只做最终评估：

    1. 创建 test_loader
    2. 加载当前默认 best model
    3. 在 test set 上预测
    4. 计算 final test accuracy / F1 / confusion matrix
    5. 保存 final_test_metrics.json
    6. 保存 final_test_confusion_matrix.png
    """
    device = torch.device("cpu")

    batch_size = 32
    validation_ratio = 0.2
    random_seed = 42
    use_augmentation = False

    print("Final test evaluation started")
    print("-" * 40)
    print(f"Model name: {DEFAULT_MODEL_NAME}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Device: {device}")

    _, _, test_loader = create_train_val_test_loaders(
        batch_size=batch_size,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
        use_augmentation=use_augmentation,
    )

    class_names = test_loader.dataset.classes

    model = load_trained_model(
        model_path=MODEL_PATH,
        device=device,
        model_name=DEFAULT_MODEL_NAME,
    )

    y_true, y_pred = collect_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    summary = create_final_test_summary(
        model_name=DEFAULT_MODEL_NAME,
        model_path=MODEL_PATH,
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    saved_metrics_path = save_final_test_metrics(
        summary=summary,
        output_path=FINAL_TEST_METRICS_PATH,
    )

    detailed_metrics = create_detailed_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    saved_confusion_matrix_path = save_confusion_matrix_plot(
        confusion_matrix_result=detailed_metrics["confusion_matrix"],
        class_names=class_names,
        output_path=FINAL_TEST_CONFUSION_MATRIX_PATH,
    )

    print_final_test_summary(summary)

    print()
    print("Saved outputs")
    print("-" * 40)
    print(f"Final test metrics: {saved_metrics_path}")
    print(f"Final test confusion matrix: {saved_confusion_matrix_path}")

    print()
    print("Final test evaluation finished")


if __name__ == "__main__":
    main()