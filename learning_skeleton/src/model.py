import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Skeleton role:
    Define the CNN model.

    Input:
    - images: [batch_size, 1, 48, 48]

    Output:
    - logits: [batch_size, 7]
    """

    def __init__(self, num_classes=7):
        super().__init__()

        # 1. CNN feature extractor: image -> feature maps
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # 2. Classifier: feature maps -> 7 emotion scores
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Main data flow inside the model
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    # Minimal check: can the model convert images into 7-class logits?
    model = SimpleCNN(num_classes=7)
    dummy_images = torch.randn(32, 1, 48, 48)
    outputs = model(dummy_images)

    print("Input shape:", dummy_images.shape)
    print("Output shape:", outputs.shape)


if __name__ == "__main__":
    main()
