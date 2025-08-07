#!/usr/bin/env python3
"""
提取exertion level标签脚本
从general_information.csv中提取每个recording对应的exertion级别(1-5)
"""

import pandas as pd
import os

def extract_exertion_labels():
    """从general_information.csv中提取exertion level标签"""
    
    # 读取元数据文件
    metadata_path = "data/general_information.csv"
    if not os.path.exists(metadata_path):
        print(f"错误: 元数据文件 {metadata_path} 不存在")
        return
    
    print("读取元数据文件...")
    df = pd.read_csv(metadata_path)
    
    # 提取Session Name和Exertion列
    labels_df = df[['Session Name', 'Exertion']].copy()
    
    # 重命名列
    labels_df.columns = ['session_id', 'exertion_level']
    
    # 检查exertion level的范围
    exertion_levels = labels_df['exertion_level'].unique()
    print(f"发现的exertion level范围: {sorted(exertion_levels)}")
    
    # 验证exertion level是否为1-5的整数
    valid_levels = [1, 2, 3, 4, 5]
    invalid_levels = [level for level in exertion_levels if level not in valid_levels]
    if invalid_levels:
        print(f"警告: 发现无效的exertion level: {invalid_levels}")
    
    # 统计每个level的数量
    print("\n各exertion level的分布:")
    level_counts = labels_df['exertion_level'].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"Level {level}: {count} 个样本")
    
    # 保存标签文件
    output_path = "data/exertion_labels.csv"
    labels_df.to_csv(output_path, index=False)
    
    print(f"\n标签提取完成!")
    print(f"总样本数: {len(labels_df)}")
    print(f"标签文件保存到: {output_path}")
    
    # 显示前几个样本
    print("\n前10个样本:")
    print(labels_df.head(10))
    
    return labels_df

def create_segment_labels():
    """为分段后的音频创建对应的标签"""
    
    # 读取exertion标签
    labels_path = "data/exertion_labels.csv"
    if not os.path.exists(labels_path):
        print(f"错误: 标签文件 {labels_path} 不存在，请先运行extract_exertion_labels()")
        return
    
    labels_df = pd.read_csv(labels_path)
    
    # 检查分段音频目录
    segmented_audio_dir = "data/segmented_audio"
    if not os.path.exists(segmented_audio_dir):
        print(f"分段音频目录 {segmented_audio_dir} 不存在")
        return
    
    # 获取所有分段音频文件
    segmented_files = [f for f in os.listdir(segmented_audio_dir) if f.endswith('.wav')]
    
    if not segmented_files:
        print("分段音频目录为空")
        return
    
    print(f"找到 {len(segmented_files)} 个分段音频文件")
    
    # 创建分段标签映射
    segment_labels = []
    
    for filename in segmented_files:
        # 解析文件名: session_id_stride_N.wav
        # 例如: d31_P01_6_0_clip_1_stride_1.wav
        session_id = filename.replace('.wav', '')
        
        # 找到对应的原始session_id (去掉_stride_N部分)
        original_session_id = None
        for label_session in labels_df['session_id']:
            if session_id.startswith(label_session):
                original_session_id = label_session
                break
        
        if original_session_id is not None:
            exertion_level = labels_df[labels_df['session_id'] == original_session_id]['exertion_level'].iloc[0]
            segment_labels.append({
                'segment_id': session_id,
                'original_session_id': original_session_id,
                'exertion_level': exertion_level
            })
        else:
            print(f"警告: 无法找到 {session_id} 对应的标签")
    
    # 保存分段标签
    segment_labels_df = pd.DataFrame(segment_labels)
    segment_labels_path = "data/segment_exertion_labels.csv"
    segment_labels_df.to_csv(segment_labels_path, index=False)
    
    print(f"\n分段标签创建完成!")
    print(f"分段标签文件保存到: {segment_labels_path}")
    print(f"成功映射的分段数: {len(segment_labels_df)}")
    
    # 显示分段标签分布
    print("\n分段标签分布:")
    level_counts = segment_labels_df['exertion_level'].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"Level {level}: {count} 个分段")
    
    return segment_labels_df

if __name__ == "__main__":
    print("=== Exertion Level 标签提取 ===")
    
    # 提取原始标签
    labels_df = extract_exertion_labels()
    
    # 创建分段标签 (基于分段后的音频)
    segment_labels_df = create_segment_labels()
    
    print("\n=== 完成 ===") 