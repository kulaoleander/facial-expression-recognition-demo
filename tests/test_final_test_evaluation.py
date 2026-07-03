import json

import pytest

from src.final_test_evaluation import (
    calculate_accuracy_from_predictions,
    create_final_test_summary,
    save_final_test_metrics,
)


def test_calculate_accuracy_from_predictions_returns_correct_accuracy():
    """
    测试 accuracy 计算是否正确。
    """
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 1, 3]

    accuracy = calculate_accuracy_from_predictions(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert accuracy == 0.75


def test_calculate_accuracy_from_predictions_rejects_empty_input():
    """
    测试 y_true 为空时是否主动报错。
    """
    with pytest.raises(ValueError):
        calculate_accuracy_from_predictions(
            y_true=[],
            y_pred=[],
        )


def test_calculate_accuracy_from_predictions_rejects_length_mismatch():
    """
    测试 y_true 和 y_pred 长度不一致时是否主动报错。
    """
    with pytest.raises(ValueError):
        calculate_accuracy_from_predictions(
            y_true=[0, 1],
            y_pred=[0],
        )


def test_create_final_test_summary_contains_expected_keys(tmp_path):
    """
    测试 final test summary 是否包含项目展示需要的核心字段。
    """
    model_path = tmp_path / "resnet18.pth"

    y_true = [0, 1, 2, 2]
    y_pred = [0, 1, 1, 2]
    class_names = ["angry", "happy", "neutral"]

    summary = create_final_test_summary(
        model_name="resnet18",
        model_path=model_path,
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    assert summary["model_name"] == "resnet18"
    assert summary["model_path"] == str(model_path)
    assert summary["num_test_samples"] == 4
    assert summary["test_accuracy"] == 0.75

    assert "macro_avg_f1" in summary
    assert "weighted_avg_f1" in summary
    assert "per_class_accuracy" in summary
    assert "classification_report" in summary
    assert "confusion_matrix" in summary

    assert isinstance(summary["confusion_matrix"], list)


def test_save_final_test_metrics_writes_json_file(tmp_path):
    """
    测试 final test metrics 是否能保存成 JSON 文件。
    """
    summary = {
        "model_name": "resnet18",
        "test_accuracy": 0.75,
    }

    output_path = tmp_path / "final_test_metrics.json"

    saved_path = save_final_test_metrics(
        summary=summary,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as file:
        loaded_data = json.load(file)

    assert loaded_data["model_name"] == "resnet18"
    assert loaded_data["test_accuracy"] == 0.75