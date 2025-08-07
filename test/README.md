# 运动强度检测项目单元测试

本目录包含运动强度检测项目的完整单元测试套件，用于验证各个模块的功能正确性。

## 测试结构

```
test/
├── README.md                    # 测试说明文档
├── test_config.yaml            # 测试配置文件
├── run_all_tests.py            # 主测试运行器
├── test_data_processing.py     # 数据处理模块测试
├── test_model.py               # 模型模块测试
├── test_training.py            # 训练模块测试
├── test_evaluation.py          # 评估模块测试
├── test_utils.py               # 工具模块测试
├── test_feature_loading.py     # 特征加载测试
├── test_progress_display.py    # 进度显示测试
├── test_wandb.py               # WandB管理测试
├── test_rtx4070_optimization.py # RTX4070优化测试
└── monitor_gpu.py              # GPU监控工具
```

## 测试模块说明

### 1. 数据处理模块测试 (`test_data_processing.py`)
- 音频数据清理功能
- 音频分割功能
- 标签提取和验证
- 特征提取（MFCC、MFB、wav2vec2）
- 音频重采样
- 完整数据处理流程

### 2. 模型模块测试 (`test_model.py`)
- VGG16模型创建和初始化
- 模型参数计算
- 前向传播测试
- 特征提取功能
- 不同配置的模型测试
- RTX4070优化模型
- 梯度流动测试
- 模型保存和加载

### 3. 训练模块测试 (`test_training.py`)
- 训练器初始化
- 优化器创建（Adam、AdamW、SGD）
- 学习率调度器
- 损失函数（交叉熵、Focal Loss）
- 训练和验证epoch
- 交叉验证数据分割
- 完整训练周期

### 4. 评估模块测试 (`test_evaluation.py`)
- 评估器初始化
- 指标计算（准确率、精确率、召回率、F1分数）
- 混淆矩阵生成
- ROC曲线绘制
- 预测分布可视化
- AUC阈值优化
- 交叉验证评估

### 5. 工具模块测试 (`test_utils.py`)
- 配置管理器
- 随机种子设置
- WandB管理器
- 错误处理
- 配置验证和合并

### 6. 其他测试模块
- **特征加载测试**: 测试数据加载器和批处理功能
- **进度显示测试**: 测试训练进度显示功能
- **WandB管理测试**: 测试实验跟踪功能
- **RTX4070优化测试**: 测试GPU优化功能

## 运行测试

### 运行所有测试
```bash
cd test
python run_all_tests.py
```

### 运行特定模块测试
```bash
# 运行数据处理测试
python run_all_tests.py --module data

# 运行模型测试
python run_all_tests.py --module model

# 运行训练测试
python run_all_tests.py --module training

# 运行评估测试
python run_all_tests.py --module evaluation

# 运行工具测试
python run_all_tests.py --module utils
```

### 运行快速测试（核心模块）
```bash
python run_all_tests.py --quick
```

### 列出所有测试模块
```bash
python run_all_tests.py --list
```

### 直接运行单个测试文件
```bash
# 运行数据处理测试
python test_data_processing.py

# 运行模型测试
python test_model.py

# 运行训练测试
python test_training.py

# 运行评估测试
python test_evaluation.py

# 运行工具测试
python test_utils.py
```

## 测试配置

测试使用 `test_config.yaml` 配置文件，包含：
- 数据配置（特征目录、标签文件等）
- 模型配置（维度、类别数等）
- 训练配置（批次大小、学习率等）
- 系统配置（随机种子、工作进程数等）
- WandB配置（实验跟踪设置）
- 测试特定配置（超时、清理等）

## 测试环境要求

### Python依赖
- Python 3.8+
- PyTorch 1.12+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PyYAML
- WandB（可选）

### 硬件要求
- CPU: 支持多核处理
- 内存: 至少4GB RAM
- GPU: 可选，用于GPU相关测试
- 存储: 至少1GB可用空间

## 测试输出

### 成功输出示例
```
============================================================
开始运行 数据处理模块 测试
============================================================
测试音频数据清理功能...
清理结果: {'cleaned_files': 3, 'removed_files': 0, 'errors': []}
测试音频文件验证功能...
音频文件验证测试通过
...
✅ 数据处理模块 测试通过 (耗时: 15.23秒)
```

### 失败输出示例
```
❌ 模型模块 测试失败 (耗时: 8.45秒)
错误输出:
Traceback (most recent call last):
  File "test_model.py", line 45, in test_model_creation
    model = VGG16ExertionModel(...)
AssertionError: 模型创建失败
```

## 测试覆盖率

当前测试套件覆盖以下功能：

- ✅ 数据处理流程（音频清理、分割、特征提取）
- ✅ 模型架构（VGG16、前向传播、参数计算）
- ✅ 训练流程（优化器、调度器、损失函数）
- ✅ 评估指标（准确率、混淆矩阵、ROC曲线）
- ✅ 工具功能（配置管理、随机种子、WandB）
- ✅ 错误处理和边界情况
- ✅ 集成测试和端到端流程

## 持续集成

测试套件设计支持持续集成环境：
- 自动化测试执行
- 测试结果报告
- 失败测试通知
- 测试覆盖率统计

## 故障排除

### 常见问题

1. **导入错误**
   - 确保在项目根目录运行测试
   - 检查Python路径设置

2. **CUDA相关错误**
   - 测试会自动检测CUDA可用性
   - 无CUDA时使用CPU进行测试

3. **内存不足**
   - 减少测试批次大小
   - 使用快速测试模式

4. **超时错误**
   - 增加测试超时时间
   - 检查系统资源使用情况

### 调试模式

启用详细输出：
```bash
python -v test_data_processing.py
```

## 贡献指南

添加新测试时请遵循以下规范：

1. **测试文件命名**: `test_<module_name>.py`
2. **测试类命名**: `Test<ClassName>`
3. **测试方法命名**: `test_<function_name>`
4. **使用临时文件**: 避免污染项目目录
5. **清理资源**: 测试完成后清理临时文件
6. **添加文档**: 为测试添加清晰的文档说明

## 更新日志

- **v1.0.0**: 初始测试套件
  - 基础功能测试
  - 数据处理测试
  - 模型架构测试
  - 训练流程测试
  - 评估指标测试
  - 工具功能测试
