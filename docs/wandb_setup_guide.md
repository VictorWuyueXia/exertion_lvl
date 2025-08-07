# Weights & Biases 配置和使用指南

## 概述

本项目已集成Weights & Biases (WandB)来监控训练过程和GPU使用情况。WandB是一个强大的实验跟踪平台，可以帮助您：

- 实时监控训练指标（损失、准确率等）
- 监控GPU使用率、内存使用和温度
- 监控系统资源（CPU、内存、磁盘）
- 可视化训练曲线和混淆矩阵
- 保存和版本控制模型
- 比较不同实验的结果

## 安装和设置

### 1. 安装依赖

```bash
pip install wandb psutil
```

### 2. 注册WandB账户

1. 访问 [https://wandb.ai](https://wandb.ai)
2. 注册一个免费账户
3. 获取API密钥

### 3. 登录WandB

```bash
wandb login
```

然后输入您的API密钥。

### 4. 验证安装

运行测试脚本验证配置：

```bash
python test/test_wandb.py
```

## 配置文件

### 主配置文件

`config/wandb_config.yaml` 包含所有WandB相关配置：

```yaml
wandb:
  # 项目名称
  project: "exertion-level-detection"
  
  # 实验名称模板
  name: "vgg16_wav2vec2_layer4_{timestamp}"
  
  # 实验标签
  tags: ["vgg16", "wav2vec2", "exertion-detection", "audio-analysis"]
  
  # 是否启用WandB
  enabled: true
  
  # 监控配置
  monitoring:
    gpu_interval: 30      # GPU监控间隔（秒）
    system_interval: 60   # 系统监控间隔（秒）
    gpu_usage: true       # 启用GPU监控
    memory_usage: true    # 启用内存监控
    cpu_usage: true       # 启用CPU监控
```

### 训练配置文件

`config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml` 包含训练参数：

```yaml
training:
  epochs: 100
  batch_size: 8
  learning_rate: 1e-4
  optimizer: "adamw"
  scheduler: "cosine"
```

## 使用方法

### 1. 开始训练

使用配置文件版本的主训练脚本：

```bash
python src/main_train_config.py --config config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml
```

### 2. 查看训练进度

1. 在终端中会显示训练进度
2. 访问 [https://wandb.ai](https://wandb.ai) 查看实时仪表板
3. 在项目页面可以看到：
   - 训练和验证指标曲线
   - GPU使用率图表
   - 系统资源监控
   - 混淆矩阵
   - 模型文件

### 3. 监控指标

WandB会自动记录以下指标：

#### 训练指标
- `train/loss`: 训练损失
- `train/accuracy`: 训练准确率
- `train/learning_rate`: 学习率
- `train/gradient_norm`: 梯度范数
- `train/parameter_norm`: 参数范数

#### 验证指标
- `val/loss`: 验证损失
- `val/accuracy`: 验证准确率

#### GPU指标
- `gpu/utilization`: GPU使用率
- `gpu/memory_allocated_gb`: 已分配GPU内存
- `gpu/memory_reserved_gb`: 已保留GPU内存
- `gpu/memory_usage_percent`: GPU内存使用百分比
- `gpu/temperature_celsius`: GPU温度（如果可用）

#### 系统指标
- `system/cpu_percent`: CPU使用率
- `system/memory_percent`: 内存使用率
- `system/disk_percent`: 磁盘使用率

#### 测试指标
- `test/mean_accuracy`: 平均准确率
- `test/std_accuracy`: 准确率标准差
- `test/best_accuracy`: 最佳准确率
- `test/worst_accuracy`: 最差准确率

## 高级功能

### 1. 自定义监控

您可以修改 `src/utils/wandb_manager.py` 来添加自定义监控指标：

```python
# 添加自定义指标
self.wandb_manager.log_training_metrics({
    'custom_metric': custom_value
}, step=epoch)
```

### 2. 模型保存

训练过程中会自动保存最佳模型到WandB：

```python
# 模型会自动保存到WandB Artifacts
self.wandb_manager.save_model(model, model_path, metadata)
```

### 3. 实验比较

在WandB界面中可以：
- 比较不同实验的结果
- 查看参数对性能的影响
- 生成报告和图表

## 故障排除

### 1. WandB连接问题

如果遇到连接问题：

```bash
# 检查网络连接
ping api.wandb.ai

# 重新登录
wandb login

# 检查API密钥
echo $WANDB_API_KEY
```

### 2. GPU监控问题

如果GPU监控不工作：

```bash
# 安装pynvml（用于GPU温度监控）
pip install pynvml

# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. 权限问题

确保有足够的权限：

```bash
# 检查文件权限
ls -la config/wandb_config.yaml

# 检查目录权限
ls -la result/
```

## 最佳实践

### 1. 实验命名

使用有意义的实验名称：

```yaml
name: "vgg16_wav2vec2_layer4_batch8_lr1e4_{timestamp}"
```

### 2. 标签使用

使用标签来组织实验：

```yaml
tags: ["vgg16", "wav2vec2", "layer4", "batch8", "lr1e4"]
```

### 3. 配置版本控制

将配置文件提交到Git：

```bash
git add config/wandb_config.yaml
git commit -m "更新WandB配置"
```

### 4. 定期检查

定期检查WandB仪表板：
- 监控训练进度
- 检查GPU使用情况
- 分析性能趋势

## 示例输出

训练开始时的输出示例：

```
WandB配置已加载
WandB初始化成功: vgg16_wav2vec2_layer4_20241201_143022
系统监控线程已启动
开始5折交叉验证训练

Fold 1/5
Epoch 1/100
训练 - Loss: 1.2345, Acc: 0.4567
验证 - Loss: 1.1234, Acc: 0.4789
学习率: 0.000100
保存最佳模型，验证准确率: 0.4789
```

WandB仪表板将显示：
- 实时训练曲线
- GPU使用率图表
- 系统资源监控
- 混淆矩阵可视化

## 联系支持

如果遇到问题：
1. 查看WandB官方文档：https://docs.wandb.ai/
2. 检查项目日志文件
3. 联系项目维护者
