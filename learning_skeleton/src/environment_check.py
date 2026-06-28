import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd
import PIL
import pytest
import sklearn
import streamlit
import torch
import torchvision


def get_environment_info():
    # 骨架作用：确认项目依赖是否能正常导入。
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pillow_version": PIL.__version__,
        "matplotlib_version": matplotlib.__version__,
        "sklearn_version": sklearn.__version__,
        "streamlit_version": streamlit.__version__,
        "pytest_version": pytest.__version__,
    }


def main():
    info = get_environment_info()

    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
