import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import SimpleCNN
from src.train import save_model, train_model, train_one_epoch


def create_dummy_train_loader(batch_size=16):
    """
    创建一个很小的假训练集，用来快速测试训练函数。

    为什么不用真实 FER2013 数据？
    - 真实数据太多，测试会变慢
    - 单元测试只需要验证训练逻辑是否正确
    """
    images = torch.randn(64, 1, 48, 48)
    labels = torch.randint(0, 7, (64,))

    dataset = TensorDataset(images, labels)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return loader


def test_train_one_epoch_returns_float_loss():
    """
    测试 train_one_epoch 是否返回一个 float 类型的 average loss。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    average_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    assert isinstance(average_loss, float)


def test_train_one_epoch_returns_positive_loss():
    """
    测试 average loss 是否是正数。

    对 CrossEntropyLoss 来说，正常情况下 loss 应该大于 0。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    average_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    assert average_loss > 0.0


def test_train_one_epoch_updates_model_parameters():
    """
    测试训练一个 epoch 后，模型参数是否发生变化。

    如果参数完全没有变化，说明 backward 或 optimizer.step 可能没有正常工作。
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 保存训练前的模型参数。
    # 这里不用列表推导式，改成普通 for 循环，方便理解。
    parameters_before_training = []

    for parameter in model.parameters():
        if parameter.requires_grad:
            copied_parameter = parameter.clone().detach()
            parameters_before_training.append(copied_parameter)

    # 运行一次训练。
    train_one_epoch(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    # 保存训练后的模型参数。
    parameters_after_training = []

    for parameter in model.parameters():
        if parameter.requires_grad:
            current_parameter = parameter.detach()
            parameters_after_training.append(current_parameter)

    # 比较训练前后参数是否发生变化。
    # 只要有一个参数变了，就说明 optimizer.step() 确实更新了模型。
    has_parameter_changed = False

    for before, after in zip(parameters_before_training, parameters_after_training):
        if not torch.equal(before, after):
            has_parameter_changed = True
            break

    assert has_parameter_changed


def test_train_model_runs_multiple_epochs():
    """
    测试 train_model 是否能完成多个 epoch 的训练流程。

    这个测试不追求 accuracy 高低，
    只验证完整流程能跑通：
    train_one_epoch -> evaluate_accuracy -> print result
    """
    device = torch.device("cpu")

    model = SimpleCNN(num_classes=7).to(device)
    train_loader = create_dummy_train_loader(batch_size=16)
    test_loader = create_dummy_train_loader(batch_size=16)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
    )


def test_save_model_creates_model_file(tmp_path):
    """
    测试 save_model 是否能把模型权重保存成 .pth 文件。

    tmp_path 是 pytest 提供的临时文件夹。
    用它测试不会污染真实 outputs/models 文件夹。
    """
    model = SimpleCNN(num_classes=7)

    model_path = tmp_path / "test_model.pth"

    save_model(
        model=model,
        model_path=model_path,
    )

    assert model_path.exists()
    assert model_path.stat().st_size > 0