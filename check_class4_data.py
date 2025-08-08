#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查类别4数据的脚本
分析为什么类别4在最终结果中缺失
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import get_session_metadata, split_train_test_val

def check_class4_in_original_data():
    """检查原始数据中类别4的情况"""
    print("=" * 60)
    print("检查原始数据中类别4的情况")
    print("=" * 60)
    
    # 读取数据
    feature_dir = "data/features"
    label_file = "data/general_information.csv"
    
    metadata_df = get_session_metadata(feature_dir)
    labels_df = pd.read_csv(label_file)
    
    print(f"总会话数: {len(metadata_df)}")
    print(f"标签记录数: {len(labels_df)}")
    
    # 检查标签文件中的类别分布
    print(f"\n标签文件中的类别分布:")
    exertion_counts = labels_df['Exertion'].value_counts().sort_index()
    for exertion_level, count in exertion_counts.items():
        print(f"  类别 {exertion_level-1}: {count} 个样本")
    
    # 检查类别4的具体情况
    class4_sessions = labels_df[labels_df['Exertion'] == 5]['Session Name'].tolist()
    print(f"\n类别4的会话数: {len(class4_sessions)}")
    
    if class4_sessions:
        print("类别4的会话:")
        for session in class4_sessions[:10]:  # 显示前10个
            print(f"  {session}")
        if len(class4_sessions) > 10:
            print(f"  ... 还有 {len(class4_sessions) - 10} 个会话")
    
    # 检查类别4的会话是否有对应的特征文件
    class4_with_features = 0
    class4_sessions_with_features = []
    
    for session_name in class4_sessions:
        # 查找所有以该会话名开头的特征目录
        session_dirs = [d for d in os.listdir(feature_dir) if d.startswith(session_name)]
        if session_dirs:
            class4_with_features += 1
            class4_sessions_with_features.extend(session_dirs)
    
    print(f"\n类别4会话中有特征文件的: {class4_with_features}/{len(class4_sessions)}")
    print(f"类别4的特征会话总数: {len(class4_sessions_with_features)}")
    
    if class4_sessions_with_features:
        print("类别4的特征会话示例:")
        for session in class4_sessions_with_features[:5]:
            print(f"  {session}")
        if len(class4_sessions_with_features) > 5:
            print(f"  ... 还有 {len(class4_sessions_with_features) - 5} 个会话")
    
    return class4_sessions_with_features

def check_class4_in_split_data(class4_sessions):
    """检查分割后类别4的情况"""
    print("\n" + "=" * 60)
    print("检查分割后类别4的情况")
    print("=" * 60)
    
    feature_dir = "data/features"
    label_file = "data/general_information.csv"
    
    metadata_df = get_session_metadata(feature_dir)
    labels_df = pd.read_csv(label_file)
    
    # 使用基于参与者的分割
    train_sessions, val_sessions, test_sessions = split_train_test_val(
        metadata_df, test_size=0.1, val_size=0.1, random_state=42
    )
    
    # 检查类别4在各集合中的分布
    def count_class4_sessions(sessions):
        count = 0
        for session_id in sessions:
            base_session_id = session_id.split('_stride_')[0] if '_stride_' in session_id else session_id
            label_row = labels_df[labels_df['Session Name'] == base_session_id]
            if not label_row.empty and label_row['Exertion'].iloc[0] == 5:
                count += 1
        return count
    
    train_class4 = count_class4_sessions(train_sessions)
    val_class4 = count_class4_sessions(val_sessions)
    test_class4 = count_class4_sessions(test_sessions)
    
    print(f"训练集中类别4会话数: {train_class4}")
    print(f"验证集中类别4会话数: {val_class4}")
    print(f"测试集中类别4会话数: {test_class4}")
    
    # 检查类别4的参与者分布
    class4_participants = set()
    for session_id in class4_sessions:
        if session_id in metadata_df['session'].values:
            participant = metadata_df[metadata_df['session'] == session_id]['participant'].iloc[0]
            class4_participants.add(participant)
    
    print(f"\n类别4涉及的参与者数: {len(class4_participants)}")
    print("类别4的参与者:")
    for participant in sorted(class4_participants):
        print(f"  {participant}")
    
    return train_class4, val_class4, test_class4

def check_class4_in_balanced_data():
    """检查平衡后类别4的情况"""
    print("\n" + "=" * 60)
    print("检查平衡后类别4的情况")
    print("=" * 60)
    
    feature_dir = "data/features"
    label_file = "data/general_information.csv"
    
    metadata_df = get_session_metadata(feature_dir)
    labels_df = pd.read_csv(label_file)
    
    # 获取训练集
    train_sessions, val_sessions, test_sessions = split_train_test_val(
        metadata_df, test_size=0.1, val_size=0.1, random_state=42
    )
    
    # 创建训练集元数据
    train_metadata = metadata_df[metadata_df['session'].isin(train_sessions)].copy()
    
    print(f"训练集原始会话数: {len(train_metadata)}")
    
    # 检查训练集中类别4的原始数量
    class4_count = 0
    for _, row in train_metadata.iterrows():
        session_id = row["session"]
        base_session_id = session_id.split('_stride_')[0] if '_stride_' in session_id else session_id
        label_row = labels_df[labels_df['Session Name'] == base_session_id]
        if not label_row.empty and label_row['Exertion'].iloc[0] == 5:
            class4_count += 1
    
    print(f"训练集中类别4原始数量: {class4_count}")
    
    if class4_count == 0:
        print("❌ 问题发现: 训练集中没有类别4的样本！")
        print("这是导致类别4在最终结果中缺失的根本原因。")
        return False
    
    return True

def main():
    """主函数"""
    print("开始检查类别4数据...")
    
    # 1. 检查原始数据中类别4的情况
    class4_sessions = check_class4_in_original_data()
    
    # 2. 检查分割后类别4的情况
    train_class4, val_class4, test_class4 = check_class4_in_split_data(class4_sessions)
    
    # 3. 检查平衡后类别4的情况
    has_class4_in_train = check_class4_in_balanced_data()
    
    print("\n" + "=" * 60)
    print("问题诊断总结")
    print("=" * 60)
    
    if not has_class4_in_train:
        print("❌ 根本问题: 训练集中没有类别4的样本")
        print("解决方案:")
        print("1. 检查数据分割策略，确保类别4的参与者被分配到训练集")
        print("2. 调整分割参数，增加训练集大小")
        print("3. 使用分层分割，确保每个类别都有代表")
    else:
        print("✅ 训练集中有类别4的样本")
        print("可能的问题:")
        print("1. 数据平衡过程中类别4被过滤")
        print("2. 特征文件缺失导致类别4样本被排除")
        print("3. 数据加载过程中的索引错误")

if __name__ == "__main__":
    main()

