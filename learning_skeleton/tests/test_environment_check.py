from learning_skeleton.src.environment_check import get_environment_info


def test_environment_info_contains_core_keys():
    # 验证环境检查函数返回核心依赖信息。
    info = get_environment_info()

    required_keys = [
        "python_version",
        "torch_version",
        "torchvision_version",
        "cuda_available",
        "opencv_version",
        "streamlit_version",
        "pytest_version",
    ]

    for key in required_keys:
        assert key in info


def test_cuda_available_is_boolean():
    info = get_environment_info()
    assert isinstance(info["cuda_available"], bool)
