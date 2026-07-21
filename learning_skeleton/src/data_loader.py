from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


def create_transforms():
    """
    Skeleton role:
    Convert raw images into tensors that the CNN can accept.

    Main idea:
    image file -> grayscale -> tensor [1, 48, 48]
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )


def create_datasets():
    """
    Skeleton role:
    Read train/test folders as PyTorch datasets.

    ImageFolder rule:
    class folder name = label name
    """
    transform = create_transforms()

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

    return train_dataset, test_dataset


def create_data_loaders(batch_size=32):
    """
    Skeleton role:
    Wrap datasets into DataLoaders so training can use mini-batches.
    """
    train_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main():
    # Minimal check: get one training batch and inspect its shape.
    train_loader, _ = create_data_loaders(batch_size=32)
    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)


if __name__ == "__main__":
    main()
