import subprocess
import torch
import tensorflow as tf
import numpy as np
import sys

# 嘗試導入 visdom 和 tensorboard
try:
    import visdom
    visdom_version = visdom.__version__
except ImportError:
    visdom_version = "Visdom not installed"

try:
    import tensorboard
    tensorboard_version = tensorboard.__version__
except ImportError:
    tensorboard_version = "TensorBoard not installed"

# 取得 nvcc (CUDA Compiler) 版本
try:
    nvcc_version = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.split("\n")[-2]
except FileNotFoundError:
    nvcc_version = "nvcc not found"

# 取得 cuDNN 版本
try:
    cudnn_version = torch.backends.cudnn.version()
except AttributeError:
    cudnn_version = "cuDNN not found"

# 取得 CUDA 版本 (來自 PyTorch)
cuda_version = torch.version.cuda if torch.cuda.is_available() else "CUDA not available"

# 取得 TensorFlow CUDA 版本
tf_cuda_version = tf.sysconfig.get_build_info().get("cuda_version", "Not found")

# 取得 TensorFlow cuDNN 版本
tf_cudnn_version = tf.sysconfig.get_build_info().get("cudnn_version", "Not found")

# 輸出所有版本資訊
print(f"nvcc Version: {nvcc_version}")
print(f"CUDA Version (PyTorch): {cuda_version}")
print(f"CUDA Version (TensorFlow): {tf_cuda_version}")
print(f"cuDNN Version (PyTorch): {cudnn_version}")
print(f"cuDNN Version (TensorFlow): {tf_cudnn_version}")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Numpy Version: {np.__version__}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Visdom Version: {visdom_version}")
print(f"TensorBoard Version: {tensorboard_version}")