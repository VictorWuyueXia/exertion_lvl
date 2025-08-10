# 运动强度检测项目 (Exertion Level Detection)

基于音频信号的运动强度检测系统，使用深度学习技术分析运动时的呼吸音频来评估运动强度。

## 项目概述

本项目旨在通过分析运动过程中的呼吸音频信号，自动检测和分类运动强度等级。系统采用wav2vec2模型提取音频特征，结合机器学习算法进行运动强度预测。

## 项目结构

```
exertion_lvl/
├── src/                    # 源代码目录
│   ├── data/              # 数据处理模块
│   │   ├── clean_data.py          # 音频数据清理
│   │   ├── segment_audio.py       # 音频分段处理
│   │   ├── extract_labels.py      # 标签提取
│   │   ├── get_features.py        # wav2vec2特征提取
│   │   ├── loader.py              # 数据加载器
│   │   ├── process_data.py        # 数据处理主流程
│   │   ├── normalizer.py          # 特征标准化
│   │   ├── normalizer_memory_efficient.py # 内存高效标准化
│   │   └── balancer.py            # 数据平衡器
│   ├── models/            # 模型定义
│   │   └── vgg16_exertion.py      # VGG16运动强度检测模型
│   ├── training/          # 训练模块
│   ├── evaluation/        # 评估模块
│   ├── utils/             # 工具模块
│   │   ├── config_manager.py      # 配置管理
│   │   ├── wandb_manager.py       # WandB管理
│   │   ├── data_leakage_checker.py # 数据泄露检查
│   │   └── set_seed.py            # 随机种子设置
│   ├── train.py           # 主训练脚本
│   ├── train_with_config.py       # 配置文件训练脚本
│   └── evaluate.py        # 结果评估脚本
├── test/                   # 测试文件
├── config/                 # 配置文件
├── data/                   # 数据目录
│   ├── audio/                     # 原始音频文件
│   ├── general_information.csv    # 基本信息文件
│   └── data_README.md             # 数据说明文档
├── result/                 # 训练结果（包含在git中）
├── scripts/                # 脚本文件
├── docs/                   # 文档
├── requirements.txt        # Python依赖包
└── README.md              # 项目说明文档
```

## 数据处理流程

1. **数据清理** (`clean_data.py`): 对原始音频进行降噪和预处理
2. **音频分段** (`segment_audio.py`): 将长音频分割成固定长度的片段
3. **标签提取** (`extract_labels.py`): 从元数据中提取运动强度标签
4. **特征提取** (`get_features.py`): 使用wav2vec2模型提取音频特征
5. **数据加载** (`loader.py`): 构建数据加载器用于模型训练

## 环境要求

- Python 3.8+
- PyTorch
- transformers (wav2vec2)
- librosa
- pandas
- numpy
- scikit-learn
- wandb (实验跟踪)
- psutil (系统监控)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 快速设置WandB（推荐）

```bash
# 运行快速设置脚本
./scripts/setup_wandb.sh

# 或者手动设置
pip install wandb psutil
wandb login
python test/test_wandb.py
```

### 2. 数据预处理

```python
from src.data.process_data import process_data_pipeline

# 运行完整的数据处理流程
process_data_pipeline()
```

### 3. 开始训练（带WandB监控）

```bash
# 使用配置文件版本（推荐）
python src/train_with_config.py --config config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml

# 或者使用基础版本
python src/train.py
```

### 4. 查看训练进度

1. 终端实时输出
2. 访问 [https://wandb.ai](https://wandb.ai) 查看可视化仪表板
3. 监控GPU使用率、系统资源和训练指标

### 5. 结果评估

```bash
# 评估最新训练结果
python src/evaluate.py
```

### 6. GPU监控

```bash
# 安装nvtop（推荐）
sudo apt install nvtop
nvtop

# 或者使用Python监控
python -c "import torch; print(f'GPU可用: {torch.cuda.is_available()}')"
```

## 技术特点

- **wav2vec2特征提取**: 使用预训练的wav2vec2模型提取高质量的音频特征
- **模块化设计**: 清晰的数据处理管道，便于维护和扩展
- **GPU优化**: 针对现代GPU进行优化
- **Weights & Biases集成**: 完整的实验跟踪和监控系统
- **实时监控**: 提供GPU使用情况、系统资源和训练指标监控
- **可视化仪表板**: 实时查看训练进度和性能分析

## 数据说明

- **音频格式**: WAV格式，采样率16kHz
- **运动强度等级**: 5-10级（5级为轻度，10级为极重度）
- **数据来源**: 运动过程中的呼吸音频记录

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 文档

- [WandB配置和使用指南](docs/wandb_setup_guide.md) - 详细的WandB设置和使用说明
- [项目结构说明](data/data_README.md) - 数据格式和处理说明

## 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱: victor.w.x@icloud.com
- 项目地址: [GitHub Repository]

---

**注意**: 本项目仅用于研究目的，请确保遵守相关数据使用协议。 