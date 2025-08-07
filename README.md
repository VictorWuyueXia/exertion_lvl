# 运动强度检测项目 (Exertion Level Detection)

基于音频信号的运动强度检测系统，使用深度学习技术分析运动时的呼吸音频来评估运动强度。

## 项目概述

本项目旨在通过分析运动过程中的呼吸音频信号，自动检测和分类运动强度等级。系统采用wav2vec2模型提取音频特征，结合机器学习算法进行运动强度预测。

## 项目结构

```
exertion_lvl/
├── src/                    # 源代码目录
│   └── data/              # 数据处理模块
│       ├── clean_data.py          # 音频数据清理
│       ├── segment_audio.py       # 音频分段处理
│       ├── extract_labels.py      # 标签提取
│       ├── get_features.py        # wav2vec2特征提取
│       ├── loader.py              # 数据加载器
│       └── process_data.py        # 数据处理主流程
├── test/                   # 测试文件
│   ├── monitor_gpu.py             # GPU监控工具
│   └── test_rtx4070_optimization.py # RTX4070优化测试
├── data/                   # 数据目录
│   ├── audio/                     # 原始音频文件
│   ├── general_information.csv    # 基本信息文件
│   └── data_README.md             # 数据说明文档
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

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据预处理

```python
from src.data.process_data import process_data_pipeline

# 运行完整的数据处理流程
process_data_pipeline()
```

### 2. GPU监控

```python
from test.monitor_gpu import monitor_gpu_usage

# 监控GPU使用情况
monitor_gpu_usage()
```

## 技术特点

- **wav2vec2特征提取**: 使用预训练的wav2vec2模型提取高质量的音频特征
- **模块化设计**: 清晰的数据处理管道，便于维护和扩展
- **GPU优化**: 针对RTX4070等现代GPU进行优化
- **实时监控**: 提供GPU使用情况监控工具

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

## 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱: victor.w.x@icloud.com
- 项目地址: [GitHub Repository]

---

**注意**: 本项目仅用于研究目的，请确保遵守相关数据使用协议。 