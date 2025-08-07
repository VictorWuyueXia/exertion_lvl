#!/bin/bash

# Weights & Biases 快速设置脚本
# 用于在WSL环境中快速配置WandB

echo "=========================================="
echo "Weights & Biases 快速设置脚本"
echo "=========================================="

# 检查Python环境
echo "检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "错误: Python未安装"
    exit 1
fi

python_version=$(python --version 2>&1)
echo "Python版本: $python_version"

# 检查pip
if ! command -v pip &> /dev/null; then
    echo "错误: pip未安装"
    exit 1
fi

# 安装依赖
echo "安装WandB依赖..."
pip install wandb psutil

# 检查安装
echo "检查WandB安装..."
python -c "import wandb; print(f'WandB版本: {wandb.__version__}')"

# 检查CUDA
echo "检查CUDA环境..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA可用: {torch.cuda.get_device_name(0)}')
    print(f'CUDA版本: {torch.version.cuda}')
else:
    print('CUDA不可用，将使用CPU训练')
"

# 检查GPU监控
echo "检查GPU监控..."
python -c "
try:
    import pynvml
    print('pynvml可用，GPU温度监控已启用')
except ImportError:
    print('pynvml未安装，建议安装以获得更好的GPU监控: pip install pynvml')
"

# 运行WandB设置
echo "运行WandB设置..."
python src/utils/wandb_setup.py

# 运行测试
echo "运行WandB功能测试..."
python test/test_wandb.py

echo "=========================================="
echo "设置完成！"
echo "=========================================="
echo "现在可以开始训练:"
echo "python src/main_train_config.py --config config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml"
echo ""
echo "访问 https://wandb.ai 查看训练进度"
