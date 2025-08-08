#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据平衡器 - 确保每个类别都有指定数量的样本
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import os

class DataBalancer:
    """数据平衡器"""
    
    def __init__(self, target_samples_per_class=300, random_seed=42, ignore_classes=None):
        """
        初始化数据平衡器
        Args:
            target_samples_per_class: 每个类别的目标样本数量
            random_seed: 随机种子
            ignore_classes: 要忽略的类别列表
        """
        self.target_samples_per_class = target_samples_per_class
        self.random_seed = random_seed
        self.ignore_classes = ignore_classes or []
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
        
        # 平衡数据
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
                # 如果样本不足，随机重复
                selected_samples = samples.copy()
                while len(selected_samples) < self.target_samples_per_class:
                    selected_samples.extend(random.sample(samples, min(len(samples), self.target_samples_per_class - len(selected_samples))))
                print(f"类别 {class_id}: 从 {current_count} 个样本扩展到 {len(selected_samples)} 个")
            
            balanced_indices.extend(selected_samples)
        
        # 创建平衡后的DataFrame
        balanced_metadata_df = metadata_df.iloc[balanced_indices].reset_index(drop=True)
        
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

def balance_audio_dataset(metadata_df, labels_df, feature_dir, target_samples_per_class=300, random_seed=42, ignore_classes=None):
    """
    平衡音频数据集的便捷函数
    Args:
        metadata_df: 元数据DataFrame
        labels_df: 标签DataFrame
        feature_dir: 特征目录
        target_samples_per_class: 每个类别的目标样本数量
        random_seed: 随机种子
        ignore_classes: 要忽略的类别列表
    Returns:
        balanced_metadata_df: 平衡后的元数据DataFrame
    """
    balancer = DataBalancer(target_samples_per_class, random_seed, ignore_classes)
    return balancer.balance_dataset(metadata_df, labels_df, feature_dir)
