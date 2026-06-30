import torch
import torch.nn as nn


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

    print("Model check")
    print("-" * 40)
    print(f"Input shape: {dummy_images.shape}")
    print(f"SimpleCNN output shape: {simple_outputs.shape}")
    print(f"ImprovedCNN output shape: {improved_outputs.shape}")

    assert simple_outputs.shape == torch.Size([32, 7])
    assert improved_outputs.shape == torch.Size([32, 7])


if __name__ == "__main__":
    main()