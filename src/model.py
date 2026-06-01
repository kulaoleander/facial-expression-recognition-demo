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

        # 特征提取部分：
        # 负责从原始 48x48 灰度图中提取图像特征
        self.features = nn.Sequential(
            # 第一组：输入 1 通道灰度图，输出 16 个特征通道
            # 输入 shape:  [batch_size, 1, 48, 48]
            # 输出 shape:  [batch_size, 16, 48, 48]
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            # 池化后宽高减半：
            # [batch_size, 16, 48, 48] -> [batch_size, 16, 24, 24]
            nn.MaxPool2d(kernel_size=2),

            # 第二组：把 16 个特征通道变成 32 个特征通道
            # 输入 shape:  [batch_size, 16, 24, 24]
            # 输出 shape:  [batch_size, 32, 24, 24]
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            # 再次池化，宽高再次减半：
            # [batch_size, 32, 24, 24] -> [batch_size, 32, 12, 12]
            nn.MaxPool2d(kernel_size=2),
        )

        # 分类部分：
        # 负责把 CNN 提取到的特征转换成 7 个类别分数
        self.classifier = nn.Sequential(
            # 把 [32, 12, 12] 拉平成 32 * 12 * 12 = 4608 个特征
            nn.Flatten(),

            # 先把 4608 个特征压缩到 128 个中间特征
            nn.Linear(32 * 12 * 12, 128),
            nn.ReLU(),

            # 输出 7 个类别分数
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


def main():
    """
    用一个假的 batch 测试模型能不能正常前向传播。

    这里不训练模型，只检查：
    输入 [32, 1, 48, 48]
    是否能输出 [32, 7]
    """
    model = SimpleCNN(num_classes=7)

    dummy_images = torch.randn(32, 1, 48, 48)
    outputs = model(dummy_images)

    print("SimpleCNN model check")
    print("-" * 40)
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {outputs.shape}")

    assert outputs.shape == torch.Size([32, 7])


if __name__ == "__main__":
    main()