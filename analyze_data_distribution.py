#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析脚本
检查数据分布情况，分析性能下降的原因
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import get_session_metadata, split_train_test_val
from src.data.balancer import balance_audio_dataset

def analyze_original_distribution():
    """分析原始数据分布"""
    print("=" * 60)
    print("分析原始数据分布")
    print("=" * 60)
    
    # 读取数据
    feature_dir = "data/features"
    label_file = "data/general_information.csv"
    
    metadata_df = get_session_metadata(feature_dir)
    labels_df = pd.read_csv(label_file)
    
    print(f"总会话数: {len(metadata_df)}")
    print(f"参与者数: {metadata_df['participant'].nunique()}")
    
    # 分析每个参与者的会话数
    participant_counts = metadata_df['participant'].value_counts()
    print(f"\n参与者会话数分布:")
    print(f"  最少: {participant_counts.min()}")
    print(f"  最多: {participant_counts.max()}")
    print(f"  平均: {participant_counts.mean():.2f}")
    
    # 分析标签分布
    class_counts = defaultdict(int)
    for _, row in metadata_df.iterrows():
        session_id = row["session"]
        base_session_id = session_id.split('_stride_')[0] if '_stride_' in session_id else session_id
        label_row = labels_df[labels_df['Session Name'] == base_session_id]
        if not label_row.empty:
            exertion_level = label_row['Exertion'].iloc[0] - 1  # 转换为0-4
            class_counts[exertion_level] += 1
    
    print(f"\n原始标签分布:")
    for class_id in sorted(class_counts.keys()):
        print(f"  类别 {class_id}: {class_counts[class_id]} 个样本")
    
    return metadata_df, labels_df, class_counts

def analyze_split_distribution(metadata_df, labels_df):
    """分析分割后的数据分布"""
    print("\n" + "=" * 60)
    print("分析分割后的数据分布")
    print("=" * 60)
    
    # 使用基于参与者的分割
    train_sessions, val_sessions, test_sessions = split_train_test_val(
        metadata_df, test_size=0.1, val_size=0.1, random_state=42
    )
    
    print(f"训练集会话数: {len(train_sessions)}")
    print(f"验证集会话数: {len(val_sessions)}")
    print(f"测试集会话数: {len(test_sessions)}")
    
    # 分析各集合的标签分布
    def get_class_distribution(sessions):
        class_counts = defaultdict(int)
        for session_id in sessions:
            base_session_id = session_id.split('_stride_')[0] if '_stride_' in session_id else session_id
            label_row = labels_df[labels_df['Session Name'] == base_session_id]
            if not label_row.empty:
                exertion_level = label_row['Exertion'].iloc[0] - 1
                class_counts[exertion_level] += 1
        return class_counts
    
    train_dist = get_class_distribution(train_sessions)
    val_dist = get_class_distribution(val_sessions)
    test_dist = get_class_distribution(test_sessions)
    
    print(f"\n训练集标签分布:")
    for class_id in sorted(train_dist.keys()):
        print(f"  类别 {class_id}: {train_dist[class_id]} 个样本")
    
    print(f"\n验证集标签分布:")
    for class_id in sorted(val_dist.keys()):
        print(f"  类别 {class_id}: {val_dist[class_id]} 个样本")
    
    print(f"\n测试集标签分布:")
    for class_id in sorted(test_dist.keys()):
        print(f"  类别 {class_id}: {test_dist[class_id]} 个样本")
    
    return train_sessions, val_sessions, test_sessions

def analyze_balanced_distribution(metadata_df, labels_df, train_sessions):
    """分析平衡后的数据分布"""
    print("\n" + "=" * 60)
    print("分析平衡后的数据分布")
    print("=" * 60)
    
    # 创建训练集元数据
    train_metadata = metadata_df[metadata_df['session'].isin(train_sessions)].copy()
    
    # 应用数据平衡
    balanced_metadata_df = balance_audio_dataset(
        train_metadata,
        labels_df,
        "data/features",
        target_samples_per_class=800,
        random_seed=42,
        ignore_classes=[],
        max_oversampling_ratio=1.5
    )
    
    print(f"平衡后总样本数: {len(balanced_metadata_df)}")
    
    # 分析平衡后的标签分布
    balanced_class_counts = defaultdict(int)
    for _, row in balanced_metadata_df.iterrows():
        session_id = row["session"]
        base_session_id = session_id.split('_stride_')[0] if '_stride_' in session_id else session_id
        label_row = labels_df[labels_df['Session Name'] == base_session_id]
        if not label_row.empty:
            exertion_level = label_row['Exertion'].iloc[0] - 1
            balanced_class_counts[exertion_level] += 1
    
    print(f"\n平衡后标签分布:")
    for class_id in sorted(balanced_class_counts.keys()):
        print(f"  类别 {class_id}: {balanced_class_counts[class_id]} 个样本")
    
    return balanced_metadata_df

def main():
    """主函数"""
    print("开始数据分析...")
    
    # 1. 分析原始分布
    metadata_df, labels_df, original_dist = analyze_original_distribution()
    
    # 2. 分析分割后分布
    train_sessions, val_sessions, test_sessions = analyze_split_distribution(metadata_df, labels_df)
    
    # 3. 分析平衡后分布
    balanced_metadata_df = analyze_balanced_distribution(metadata_df, labels_df, train_sessions)
    
    print("\n" + "=" * 60)
    print("分析总结")
    print("=" * 60)
    
    # 检查是否有类别在训练集中缺失
    train_dist = defaultdict(int)
    for session_id in train_sessions:
        base_session_id = session_id.split('_stride_')[0] if '_stride_' in session_id else session_id
        label_row = labels_df[labels_df['Session Name'] == base_session_id]
        if not label_row.empty:
            exertion_level = label_row['Exertion'].iloc[0] - 1
            train_dist[exertion_level] += 1
    
    missing_classes = []
    for class_id in range(5):
        if class_id not in train_dist or train_dist[class_id] == 0:
            missing_classes.append(class_id)
    
    if missing_classes:
        print(f"⚠️  警告: 以下类别在训练集中缺失: {missing_classes}")
    else:
        print("✅ 所有类别在训练集中都有样本")
    
    # 检查数据泄露
    train_base_sessions = set()
    val_base_sessions = set()
    
    for session_id in train_sessions:
        base_session = session_id.split('_stride_')[0] if '_stride_' in session_id else session_id
        train_base_sessions.add(base_session)
    
    for session_id in val_sessions:
        base_session = session_id.split('_stride_')[0] if '_stride_' in session_id else session_id
        val_base_sessions.add(base_session)
    
    overlap = train_base_sessions & val_base_sessions
    if overlap:
        print(f"❌ 数据泄露: 训练集和验证集存在 {len(overlap)} 个相同的基础会话")
        print(f"   重叠会话: {list(overlap)[:5]}...")
    else:
        print("✅ 无数据泄露: 训练集和验证集无重叠会话")

if __name__ == "__main__":
    main()
