#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据平衡器 - 确保每个类别都有指定数量的样本
修正版本：避免过度重复采样，确保合理分布
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import os

class DataBalancer:
    """数据平衡器"""
    
    def __init__(self, target_samples_per_class=300, random_seed=42, ignore_classes=None, max_oversampling_ratio=2.0):
        """
        初始化数据平衡器
        Args:
            target_samples_per_class: 每个类别的目标样本数量
            random_seed: 随机种子
            ignore_classes: 要忽略的类别列表
            max_oversampling_ratio: 最大过采样比例，防止过度重复
        """
        self.target_samples_per_class = target_samples_per_class
        self.random_seed = random_seed
        self.ignore_classes = ignore_classes or []
        self.max_oversampling_ratio = max_oversampling_ratio
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def balance_dataset(self, metadata_df, labels_df, feature_dir):
        """
        平衡数据集
        Args:
            metadata_df: 元数据DataFrame
            labels_df: 标签DataFrame
            feature_dir: 特征目录
        Returns:
            balanced_metadata_df: 平衡后的元数据DataFrame
        """
        print(f"开始数据平衡，目标每类别 {self.target_samples_per_class} 个样本...")
        
        # 收集每个类别的样本
        class_samples = defaultdict(list)
        
        for idx, row in metadata_df.iterrows():
            session_id = row["session"]
            session_feature_dir = os.path.join(feature_dir, session_id)
            
            # 检查特征文件是否存在
            mfcc_path = os.path.join(session_feature_dir, "mfcc.npy")
            wav2vec2_path = os.path.join(session_feature_dir, f"wav2vec2/wav2vec2_layer4.npy")
            
            if not (os.path.exists(mfcc_path) and os.path.exists(wav2vec2_path)):
                continue
            
            # 获取标签
            base_session_id = session_id.split('_stride_')[0] if '_stride_' in session_id else session_id
            label_row = labels_df[labels_df['Session Name'] == base_session_id]
            
            if not label_row.empty:
                exertion_level = label_row['Exertion'].iloc[0] - 1  # 转换为0-4
                class_samples[exertion_level].append(idx)
        
        # 打印原始分布
        print("\n原始数据分布:")
        for class_id in sorted(class_samples.keys()):
            count = len(class_samples[class_id])
            print(f"  类别 {class_id}: {count} 个样本")
        
        # 平衡数据（修正版本）
        balanced_indices = []
        
        for class_id in sorted(class_samples.keys()):
            # 跳过忽略的类别
            if class_id in self.ignore_classes:
                print(f"类别 {class_id}: 被忽略")
                continue
                
            samples = class_samples[class_id]
            current_count = len(samples)
            
            if current_count >= self.target_samples_per_class:
                # 如果样本过多，随机选择
                selected_samples = random.sample(samples, self.target_samples_per_class)
                print(f"类别 {class_id}: 从 {current_count} 个样本中选择 {self.target_samples_per_class} 个")
            else:
                # 如果样本不足，计算最大允许的重复次数
                max_allowed_samples = min(
                    self.target_samples_per_class,
                    int(current_count * self.max_oversampling_ratio)
                )
                
                if max_allowed_samples <= current_count:
                    # 如果原始样本数已经足够，直接使用
                    selected_samples = samples.copy()
                    print(f"类别 {class_id}: 使用全部 {current_count} 个样本（不超过过采样限制）")
                else:
                    # 适度重复采样
                    selected_samples = samples.copy()
                    # 计算需要重复的次数
                    repeat_times = max_allowed_samples // current_count
                    remainder = max_allowed_samples % current_count
                    
                    # 重复采样
                    for _ in range(repeat_times - 1):
                        selected_samples.extend(samples)
                    
                    # 添加剩余的随机样本
                    if remainder > 0:
                        selected_samples.extend(random.sample(samples, remainder))
                    
                    print(f"类别 {class_id}: 从 {current_count} 个样本扩展到 {len(selected_samples)} 个（适度重复）")
            
            balanced_indices.extend(selected_samples)
        
        # 创建平衡后的DataFrame
        # 确保索引在有效范围内
        valid_indices = [idx for idx in balanced_indices if idx < len(metadata_df)]
        if len(valid_indices) != len(balanced_indices):
            print(f"警告: {len(balanced_indices) - len(valid_indices)} 个索引超出范围，已过滤")
        
        balanced_metadata_df = metadata_df.iloc[valid_indices].reset_index(drop=True)
        
        # 验证平衡结果
        print(f"\n平衡后数据分布:")
        balanced_class_counts = defaultdict(int)
        for idx in balanced_indices:
            session_id = metadata_df.iloc[idx]["session"]
            base_session_id = session_id.split('_stride_')[0] if '_stride_' in session_id else session_id
            label_row = labels_df[labels_df['Session Name'] == base_session_id]
            if not label_row.empty:
                exertion_level = label_row['Exertion'].iloc[0] - 1
                balanced_class_counts[exertion_level] += 1
        
        for class_id in sorted(balanced_class_counts.keys()):
            count = balanced_class_counts[class_id]
            print(f"  类别 {class_id}: {count} 个样本")
        
        print(f"\n数据平衡完成！总样本数: {len(balanced_metadata_df)}")
        
        return balanced_metadata_df

def balance_audio_dataset(metadata_df, labels_df, feature_dir, target_samples_per_class=300, random_seed=42, ignore_classes=None, max_oversampling_ratio=2.0):
    """
    平衡音频数据集的便捷函数
    Args:
        metadata_df: 元数据DataFrame
        labels_df: 标签DataFrame
        feature_dir: 特征目录
        target_samples_per_class: 每个类别的目标样本数量
        random_seed: 随机种子
        ignore_classes: 要忽略的类别列表
        max_oversampling_ratio: 最大过采样比例
    Returns:
        balanced_metadata_df: 平衡后的元数据DataFrame
    """
    balancer = DataBalancer(target_samples_per_class, random_seed, ignore_classes, max_oversampling_ratio)
    return balancer.balance_dataset(metadata_df, labels_df, feature_dir)
