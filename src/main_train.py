#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运动强度检测模型主训练脚本
整合数据处理、模型训练和评估的完整流程
"""

import os
import sys
import json
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import AudioFeatureDataset, get_session_metadata
from src.training.trainer import ExertionTrainer
from src.evaluation.evaluator import ExertionEvaluator
from src.models.vgg16_exertion import create_model, count_parameters

def create_training_config():
    """创建训练配置"""
    config = {
        # 数据配置
        'feature_dir': 'data/features',
        'use_acoustic': True,
        'use_mfb': True,
        'use_mfcc': True,
        'use_wav2vec2': True,
        'wav2vec2_layers': [4],  # 使用第4层
        
        # 模型配置
        'mfb_dim': 40,
        'mfcc_dim': 13,
        'wav2vec2_dim': 768,
        'num_classes': 5,
        'dropout_rate': 0.5,
        'optimize_for_rtx4070': True,
        
        # 训练配置
        'batch_size': 8,  # RTX4070优化
        'epochs': 5,  # 测试用5个epoch
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'criterion': 'cross_entropy',
        'use_amp': True,  # 混合精度训练
        'patience': 20,
        
        # 系统配置
        'seed': 42,
        'num_workers': 4,
        'pin_memory': True,
        
        # 交叉验证配置
        'n_folds': 5,
        'test_size': 0.1,  # 预留10%作为测试集
    }
    
    return config

def check_data_availability():
    """检查数据可用性"""
    print("检查数据可用性...")
    
    # 检查音频文件
    audio_dir = "data/audio"
    if not os.path.exists(audio_dir):
        print(f"错误: 音频目录不存在: {audio_dir}")
        return False
    
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if len(audio_files) == 0:
        print(f"错误: 音频目录中没有找到WAV文件: {audio_dir}")
        return False
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 检查标签文件
    label_file = "data/general_information.csv"
    if not os.path.exists(label_file):
        print(f"错误: 标签文件不存在: {label_file}")
        return False
    
    labels_df = pd.read_csv(label_file)
    print(f"找到 {len(labels_df)} 个标签记录")
    
    # 检查特征文件
    feature_dir = "data/features"
    if not os.path.exists(feature_dir):
        print("特征目录不存在，需要先生成特征文件")
        return False
    
    feature_files = []
    for root, dirs, files in os.walk(feature_dir):
        feature_files.extend([os.path.join(root, f) for f in files if f.endswith('.npy')])
    
    if len(feature_files) == 0:
        print("特征目录中没有找到特征文件，需要先生成特征文件")
        return False
    
    print(f"找到 {len(feature_files)} 个特征文件")
    
    return True

def generate_features_if_needed(config):
    """如果需要，生成特征文件"""
    feature_dir = config['feature_dir']
    
    if not os.path.exists(feature_dir):
        print("特征目录不存在，开始生成特征文件...")
        os.makedirs(feature_dir, exist_ok=True)
        
        # 运行特征提取
        from src.data.get_features import generate_feature_dir_with_labels
        
        # 读取标签数据
        labels_df = pd.read_csv("data/general_information.csv")
        
        # 生成特征
        generate_feature_dir_with_labels(
            audio_dir="data/audio",
            features_dir=feature_dir,
            labels_df=labels_df,
            sr_target=16000,
            sr_feature=20,
            selected_layers=config['wav2vec2_layers'],
            batch_processing=True
        )
        
        print("特征文件生成完成")
    else:
        print("特征目录已存在，跳过特征生成")

def prepare_dataset(config):
    """准备数据集"""
    print("准备数据集...")
    
    # 处理嵌套配置结构
    if 'data' in config:
        feature_dir = config['data']['feature_dir']
        use_mfb = config['data']['features']['use_mfb']
        use_mfcc = config['data']['features']['use_mfcc']
        use_wav2vec2 = config['data']['features']['use_wav2vec2']
        wav2vec2_layers = config['data']['wav2vec2_layers']
    else:
        feature_dir = config['feature_dir']
        use_mfb = config['use_mfb']
        use_mfcc = config['use_mfcc']
        use_wav2vec2 = config['use_wav2vec2']
        wav2vec2_layers = config['wav2vec2_layers']
    
    # 获取会话元数据
    metadata_df = get_session_metadata(feature_dir)
    print(f"找到 {len(metadata_df)} 个会话")
    
    # 读取标签数据
    labels_df = pd.read_csv("data/general_information.csv")
    labels_df['segment_id'] = labels_df['Session Name']
    print(f"找到 {len(labels_df)} 个标签记录")
    
    # 创建数据集
    dataset = AudioFeatureDataset(
        metadata_df=metadata_df,
        feature_dir=feature_dir,
        labels_df=labels_df,
        use_acoustic=use_mfcc,
        use_mfb=use_mfb,
        use_embed=use_wav2vec2,
        selected_wav2vec2_layers=wav2vec2_layers
    )
    
    print(f"数据集创建完成，包含 {len(dataset)} 个样本")
    
    return dataset

def train_model(config, dataset):
    """训练模型"""
    print("开始模型训练...")
    
    # 处理嵌套配置结构
    if 'training' in config:
        training_config = config['training']
        model_config = config['model']
        system_config = config['system']
        
        # 合并配置
        merged_config = {
            **config,
            **training_config,
            **model_config,
            **system_config
        }
    else:
        merged_config = config
    
    # 创建训练器
    trainer = ExertionTrainer(merged_config)
    
    # 保存配置
    trainer._save_config()
    
    # 创建模型并显示参数数量
    model = create_model(merged_config)
    param_info = count_parameters(model)
    print(f"模型参数数量: {param_info['total_params_millions']:.2f}M")
    print(f"可训练参数数量: {param_info['trainable_params_millions']:.2f}M")
    
    # 交叉验证训练
    cv_results = trainer.cross_validation_train(dataset, merged_config)
    
    return trainer, cv_results

def evaluate_model(trainer, cv_results, config):
    """评估模型"""
    print("开始模型评估...")
    
    # 创建评估器
    evaluator = ExertionEvaluator(trainer.result_dir, config)
    
    # 评估交叉验证结果
    overall_results = evaluator.evaluate_cross_validation(cv_results)
    
    # 打印主要结果
    print("\n" + "="*50)
    print("训练完成！主要结果:")
    print("="*50)
    print(f"平均准确率: {overall_results['avg_accuracy']:.4f} ± {overall_results['std_accuracy']:.4f}")
    print(f"各Fold准确率: {[f'{acc:.4f}' for acc in overall_results['fold_accuracies']]}")
    print(f"整体F1分数: {overall_results['overall_metrics']['f1_macro']:.4f}")
    print(f"整体AUC: {overall_results['overall_metrics']['auc']:.4f}")
    print(f"结果保存目录: {trainer.result_dir}")
    print("="*50)
    
    return overall_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运动强度检测模型训练')
    parser.add_argument('--skip-feature-generation', action='store_true',
                       help='跳过特征生成步骤')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    args = parser.parse_args()
    
    print("运动强度检测模型训练开始")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 创建配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        print(f"从文件加载配置: {args.config}")
    else:
        config = create_training_config()
        print("使用默认配置")
    
    # 检查数据可用性
    if not check_data_availability():
        print("数据检查失败，退出训练")
        return
    
    # 生成特征文件（如果需要）
    if not args.skip_feature_generation:
        generate_features_if_needed(config)
    else:
        print("跳过特征生成步骤")
    
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
    
    print("\n训练流程完成！")
    print(f"所有结果已保存到: {trainer.result_dir}")

if __name__ == "__main__":
    main() 