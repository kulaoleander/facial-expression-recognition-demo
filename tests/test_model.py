import torch

from src.model import ImprovedCNN, ResNet18ExpressionModel, SimpleCNN


def test_simple_cnn_output_shape():
    """
    测试 SimpleCNN 是否能把一批 48x48 灰度图转换成 7 类输出。

    输入 shape:
    [batch_size, channels, height, width] = [32, 1, 48, 48]

    输出 shape:
    [batch_size, num_classes] = [32, 7]
    """
    model = SimpleCNN(num_classes=7)

    dummy_images = torch.randn(32, 1, 48, 48)
    outputs = model(dummy_images)

    assert outputs.shape == torch.Size([32, 7])


def test_simple_cnn_works_with_different_batch_size():
    """
    测试模型是否能处理不同 batch size。

    真实训练时，最后一个 batch 可能不是 32 张图片。
    所以模型不能只对 batch_size=32 有效。
    """
    model = SimpleCNN(num_classes=7)

    dummy_images = torch.randn(8, 1, 48, 48)
    outputs = model(dummy_images)

    assert outputs.shape == torch.Size([8, 7])


def test_simple_cnn_output_dtype_is_float32():
    """
    测试模型输出是否是 float32。

    分类模型输出的是 logits，也就是每个类别的分数。
    这些分数应该是浮点数，而不是整数标签。
    """
    model = SimpleCNN(num_classes=7)

    dummy_images = torch.randn(32, 1, 48, 48)
    outputs = model(dummy_images)

    assert outputs.dtype == torch.float32


def test_simple_cnn_has_trainable_parameters():
    """
    测试模型是否有可训练参数。

    Conv2d 和 Linear 层都应该有参数。
    如果参数数量是 0，说明模型结构肯定有问题。
    """
    model = SimpleCNN(num_classes=7)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) > 0


def test_improved_cnn_output_shape():
    """
    测试 ImprovedCNN 是否能把一批 48x48 灰度图转换成 7 类输出。

    输入 shape:
    [32, 1, 48, 48]

    输出 shape:
    [32, 7]
    """
    model = ImprovedCNN(num_classes=7)

    dummy_images = torch.randn(32, 1, 48, 48)
    outputs = model(dummy_images)

    assert outputs.shape == torch.Size([32, 7])


def test_improved_cnn_works_with_different_batch_size():
    """
    测试 ImprovedCNN 是否能处理不同 batch size。

    真实训练时，最后一个 batch 可能不是 32。
    """
    model = ImprovedCNN(num_classes=7)

    dummy_images = torch.randn(8, 1, 48, 48)
    outputs = model(dummy_images)

    assert outputs.shape == torch.Size([8, 7])


def test_improved_cnn_output_dtype_is_float32():
    """
    测试 ImprovedCNN 输出是否是 float32。

    分类模型输出 logits，应该是浮点数。
    """
    model = ImprovedCNN(num_classes=7)

    dummy_images = torch.randn(32, 1, 48, 48)
    outputs = model(dummy_images)

    assert outputs.dtype == torch.float32


def test_improved_cnn_has_trainable_parameters():
    """
    测试 ImprovedCNN 是否有可训练参数。

    Conv2d、BatchNorm2d、Linear 都应该有可训练参数。
    """
    model = ImprovedCNN(num_classes=7)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) > 0


def test_resnet18_expression_model_output_shape():
    """
    测试 ResNet18ExpressionModel 是否能输出 7 类 logits。

    注意：
    这里 use_pretrained=False，
    避免单元测试时下载预训练权重。
    """
    model = ResNet18ExpressionModel(
        num_classes=7,
        use_pretrained=False,
    )

    dummy_images = torch.randn(32, 1, 48, 48)
    outputs = model(dummy_images)

    assert outputs.shape == torch.Size([32, 7])


def test_resnet18_expression_model_works_with_different_batch_size():
    """
    测试 ResNet18ExpressionModel 是否支持不同 batch size。
    """
    model = ResNet18ExpressionModel(
        num_classes=7,
        use_pretrained=False,
    )

    dummy_images = torch.randn(8, 1, 48, 48)
    outputs = model(dummy_images)

    assert outputs.shape == torch.Size([8, 7])


def test_resnet18_expression_model_accepts_grayscale_input():
    """
    测试 ResNet18ExpressionModel 是否真的支持 1 通道灰度输入。

    如果第一层 conv1 没有从 3 通道改成 1 通道，
    这里会直接报 shape mismatch。
    """
    model = ResNet18ExpressionModel(
        num_classes=7,
        use_pretrained=False,
    )

    dummy_images = torch.randn(4, 1, 48, 48)
    outputs = model(dummy_images)

    assert outputs.shape == torch.Size([4, 7])


def test_resnet18_expression_model_has_trainable_parameters():
    """
    测试 ResNet18ExpressionModel 是否有可训练参数。

    当前阶段先保持整个 ResNet18 可训练。
    后面 fine-tuning 阶段再讨论是否冻结部分层。
    """
    model = ResNet18ExpressionModel(
        num_classes=7,
        use_pretrained=False,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) > 0


def test_resnet18_expression_model_first_conv_accepts_one_channel():
    """
    测试 ResNet18 的第一层卷积是否已经改成 1 通道输入。

    原始 ResNet18 是 3 通道 RGB 输入。
    当前项目 FER2013 是 1 通道灰度图。
    所以 conv1.in_channels 应该等于 1。
    """
    model = ResNet18ExpressionModel(
        num_classes=7,
        use_pretrained=False,
    )

    assert model.model.conv1.in_channels == 1