#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运动强度检测模型主训练脚本（配置文件版本）
使用YAML配置文件统一管理所有训练参数
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_manager import load_config_from_args, ConfigManager
from src.data.loader import AudioFeatureDataset, get_session_metadata
from src.training.trainer import ExertionTrainer
from src.evaluation.evaluator import ExertionEvaluator
from src.models.vgg16_exertion import create_model, count_parameters

def setup_environment(config: ConfigManager):
    """设置训练环境"""
    print("设置训练环境...")
    
    # 设置随机种子
    seed = config.get('system.seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 设置GPU
    device = config.get('system.gpu.device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 设置GPU内存使用比例
        memory_fraction = config.get('system.gpu.memory_fraction', 0.95)
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # 启用GPU优化
        if config.get('system.gpu.enable_benchmark', True):
            torch.backends.cudnn.benchmark = True
        
        if config.get('system.gpu.enable_flash_attention', True):
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
        
        # 启用TF32（提高计算效率）
        if config.get('system.gpu.enable_tf32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 启用自动混合精度
        if config.get('system.gpu.enable_amp', True):
            print("启用自动混合精度训练")
        
        # 启用torch.compile
        if config.get('system.gpu.enable_compile', False):
            print("启用torch.compile优化")
        
        print(f"GPU内存使用比例: {memory_fraction*100:.0f}%")
        print(f"cuDNN benchmark: {config.get('system.gpu.enable_benchmark', True)}")
        print(f"TF32: {config.get('system.gpu.enable_tf32', True)}")
        print(f"Flash Attention: {config.get('system.gpu.enable_flash_attention', True)}")
    
    return device

def check_data_availability(config: ConfigManager):
    """检查数据可用性"""
    print("检查数据可用性...")
    
    # 检查特征目录
    feature_dir = config.get('data.feature_dir')
    if not os.path.exists(feature_dir):
        print(f"错误: 特征目录不存在: {feature_dir}")
        return False
    
    # 检查特征文件
    feature_dirs = [f for f in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, f))]
    if len(feature_dirs) == 0:
        print(f"错误: 特征目录中没有找到特征文件: {feature_dir}")
        return False
    
    print(f"找到 {len(feature_dirs)} 个特征目录")
    
    # 检查标签文件
    label_file = config.get('data.label_file')
    if not os.path.exists(label_file):
        print(f"错误: 标签文件不存在: {label_file}")
        return False
    
    labels_df = pd.read_csv(label_file)
    print(f"找到 {len(labels_df)} 个标签记录")
    
    return True

def prepare_dataset(config: ConfigManager):
    """准备数据集"""
    print("准备数据集...")
    
    # 获取会话元数据
    feature_dir = config.get('data.feature_dir')
    metadata_df = get_session_metadata(feature_dir)
    print(f"找到 {len(metadata_df)} 个会话")
    
    # 读取标签数据
    label_file = config.get('data.label_file')
    labels_df = pd.read_csv(label_file)
    labels_df['segment_id'] = labels_df['Session Name']
    print(f"找到 {len(labels_df)} 个标签记录")
    
    # 创建数据集
    dataset = AudioFeatureDataset(
        metadata_df=metadata_df,
        feature_dir=feature_dir,
        labels_df=labels_df,
        use_acoustic=config.get('data.features.use_acoustic', True),
        use_mfb=config.get('data.features.use_mfb', False),
        use_embed=config.get('data.features.use_wav2vec2', True),
        selected_wav2vec2_layers=config.get('data.features.wav2vec2_layers', [4])
    )
    
    print(f"数据集创建完成，包含 {len(dataset)} 个样本")
    
    return dataset

def train_model(config: ConfigManager, dataset):
    """训练模型"""
    print("开始模型训练...")
    
    # 获取完整配置（包含数据和训练配置）
    full_config = config.get_full_config()
    
    # 创建训练器
    trainer = ExertionTrainer(full_config)
    
    # 保存配置
    config.save_config(os.path.join(trainer.result_dir, "config.yaml"))
    
    # 创建模型并显示参数数量
    model_config = config.get_model_config()
    model = create_model(model_config)
    param_info = count_parameters(model)
    print(f"模型参数数量: {param_info['total_params_millions']:.2f}M")
    print(f"可训练参数数量: {param_info['trainable_params_millions']:.2f}M")
    
    # 5折交叉验证训练
    cv_results = trainer.cross_validation_train(dataset, full_config)
    
    return trainer, cv_results

def evaluate_model(trainer, cv_results, config: ConfigManager):
    """评估模型"""
    print("开始模型评估...")
    
    # 获取评估配置
    evaluation_config = config.get_evaluation_config()
    
    # 创建评估器
    evaluator = ExertionEvaluator(trainer.result_dir, config.get_training_config())
    
    # 评估交叉验证结果
    overall_results = evaluator.evaluate_cross_validation(cv_results)
    
    # 打印主要结果
    print("\n" + "="*50)
    print("训练完成！主要结果:")
    print("="*50)
    print(f"平均准确率: {overall_results['avg_accuracy']:.4f} ± {overall_results['std_accuracy']:.4f}")
    print(f"各Fold准确率: {[f'{acc:.4f}' for acc in overall_results['fold_accuracies']]}")
    print(f"整体F1分数: {overall_results['overall_metrics']['f1_macro']:.4f}")
    
    # 安全处理AUC值
    auc_value = overall_results['overall_metrics']['auc']
    if auc_value is not None:
        print(f"整体AUC: {auc_value:.4f}")
    else:
        print("整体AUC: 无法计算")
    
    print(f"结果保存目录: {trainer.result_dir}")
    print("="*50)
    
    return overall_results

def main():
    """主函数"""
    print("运动强度检测模型训练开始（配置文件版本）")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    config = load_config_from_args()
    
    # 打印配置信息
    config.print_config()
    
    # 设置环境
    device = setup_environment(config)
    
    # 检查数据可用性
    if not check_data_availability(config):
        print("数据检查失败，退出训练")
        return
    
    # 准备数据集
    try:
        dataset = prepare_dataset(config)
    except Exception as e:
        print(f"数据集准备失败: {e}")
        return
    
    # 训练模型
    try:
        trainer, cv_results = train_model(config, dataset)
    except Exception as e:
        print(f"模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 评估模型
    try:
        overall_results = evaluate_model(trainer, cv_results, config)
    except Exception as e:
        print(f"模型评估失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 结束训练器，清理WandB资源
    try:
        trainer.finish()
    except Exception as e:
        print(f"训练器结束失败: {e}")
    
    print("\n训练流程完成！")
    print(f"所有结果已保存到: {trainer.result_dir}")

if __name__ == "__main__":
    main()
