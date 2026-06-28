import torch

from learning_skeleton.src.model import SimpleCNN


def test_model_output_shape():
    # 验证 CNN 主线：图片 batch -> 7 类 logits。
    model = SimpleCNN(num_classes=7)
    dummy_images = torch.randn(32, 1, 48, 48)

    outputs = model(dummy_images)

    assert outputs.shape == torch.Size([32, 7])


def test_model_has_trainable_parameters():
    # 验证模型里确实有需要 optimizer 更新的参数。
    model = SimpleCNN(num_classes=7)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    assert len(trainable_parameters) > 0
