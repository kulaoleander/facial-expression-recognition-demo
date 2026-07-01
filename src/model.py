import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SimpleCNN(nn.Module):
    """
    一个简单的 CNN baseline 模型。

    输入：
    - images shape: [batch_size, 1, 48, 48]

    输出：
    - logits shape: [batch_size, 7]

    这里的 7 对应 7 个表情类别：
    angry, disgust, fear, happy, neutral, sad, surprise
    """

    def __init__(self, num_classes=7):
        """
        初始化模型结构。

        num_classes:
        - 输出类别数量
        - 当前项目是 7 类表情分类，所以默认是 7
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        定义数据从输入到输出的流动过程。

        x:
        - 输入图片 batch
        - shape: [batch_size, 1, 48, 48]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImprovedCNN(nn.Module):
    """
    一个比 SimpleCNN 更强的 CNN 模型。

    它仍然是 CNN 分类模型，但比 SimpleCNN 多了：
    1. 更多卷积通道
    2. 更多卷积块
    3. BatchNorm，让训练更稳定
    4. Dropout，减少过拟合风险

    输入：
    - images shape: [batch_size, 1, 48, 48]

    输出：
    - logits shape: [batch_size, 7]
    """

    def __init__(self, num_classes=7):
        """
        初始化 ImprovedCNN 模型结构。

        num_classes:
        - 输出类别数量
        - 当前项目是 7 类表情分类，所以默认是 7
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        定义 ImprovedCNN 的前向传播。

        输入：
        - x shape: [batch_size, 1, 48, 48]

        输出：
        - logits shape: [batch_size, 7]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet18ExpressionModel(nn.Module):
    """
    基于 ResNet18 的表情识别模型。

    这个模型在项目主线中的作用：
    - SimpleCNN 是 baseline
    - ImprovedCNN 是手写增强版 CNN
    - ResNet18ExpressionModel 是 transfer learning 阶段的模型

    输入：
    - images shape: [batch_size, 1, 48, 48]

    输出：
    - logits shape: [batch_size, 7]

    为什么要改 ResNet18？
    1. 原始 ResNet18 通常输入 3 通道 RGB 图片
    2. FER2013 是 1 通道灰度图
    3. 所以要把第一层 conv1 从 3 通道改成 1 通道
    4. 原始 ResNet18 最后一层输出 ImageNet 类别
    5. 所以要把最后一层 fc 改成输出 7 类表情
    """

    def __init__(self, num_classes=7, use_pretrained=False):
        """
        初始化 ResNet18 表情识别模型。

        num_classes:
        - 输出类别数量
        - 当前项目是 7 类表情分类

        use_pretrained:
        - False：不下载预训练权重，适合测试和离线环境
        - True：使用 ImageNet 预训练权重，适合后面正式 transfer learning
        """
        super().__init__()

        if use_pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.model = resnet18(weights=weights)

        original_conv1 = self.model.conv1

        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )

        if use_pretrained:
            with torch.no_grad():
                self.model.conv1.weight.copy_(
                    original_conv1.weight.mean(dim=1, keepdim=True)
                )

        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=num_classes,
        )

    def forward(self, x):
        """
        定义 ResNet18ExpressionModel 的前向传播。

        输入：
        - x shape: [batch_size, 1, 48, 48]

        输出：
        - logits shape: [batch_size, 7]
        """
        x = self.model(x)
        return x


def main():
    """
    用假的 batch 测试模型能不能正常前向传播。

    这里不训练模型，只检查：
    输入 [32, 1, 48, 48]
    是否能输出 [32, 7]
    """
    dummy_images = torch.randn(32, 1, 48, 48)

    simple_model = SimpleCNN(num_classes=7)
    simple_outputs = simple_model(dummy_images)

    improved_model = ImprovedCNN(num_classes=7)
    improved_outputs = improved_model(dummy_images)

    resnet18_model = ResNet18ExpressionModel(
        num_classes=7,
        use_pretrained=False,
    )
    resnet18_outputs = resnet18_model(dummy_images)

    print("Model check")
    print("-" * 40)
    print(f"Input shape: {dummy_images.shape}")
    print(f"SimpleCNN output shape: {simple_outputs.shape}")
    print(f"ImprovedCNN output shape: {improved_outputs.shape}")
    print(f"ResNet18ExpressionModel output shape: {resnet18_outputs.shape}")

    assert simple_outputs.shape == torch.Size([32, 7])
    assert improved_outputs.shape == torch.Size([32, 7])
    assert resnet18_outputs.shape == torch.Size([32, 7])


if __name__ == "__main__":
    main()