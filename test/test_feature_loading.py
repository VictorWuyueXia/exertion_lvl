#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特征加载和传递
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from src.data.loader import AudioFeatureDataset, get_session_metadata
from src.models.vgg16_exertion import create_model
from src.utils.config_manager import ConfigManager

def test_feature_loading():
    """测试特征加载"""
    print("=== 测试特征加载 ===")
    
    try:
        # 加载配置
        config = ConfigManager('../config/vgg16_wav2vec2_layer4_mfcc_exertion.yaml')
        
        # 获取数据路径（相对于项目根目录）
        feature_dir = os.path.join('..', config.get('data.feature_dir'))
        label_file = os.path.join('..', config.get('data.label_file'))
        
        # 获取会话元数据
        metadata_df = get_session_metadata(feature_dir)
        print(f"找到 {len(metadata_df)} 个会话")
        
        # 读取标签数据
        import pandas as pd
        labels_df = pd.read_csv(label_file)
        print(f"找到 {len(labels_df)} 个标签记录")
        
        # 创建数据集
        dataset = AudioFeatureDataset(
            metadata_df=metadata_df,
            feature_dir=feature_dir,
            labels_df=labels_df,
            use_acoustic=config.get('data.features.use_acoustic'),
            use_mfb=config.get('data.features.use_mfb'),
            use_embed=config.get('data.features.use_wav2vec2'),
            selected_wav2vec2_layers=config.get('data.features.wav2vec2_layers')
        )
        
        print(f"数据集创建完成，包含 {len(dataset)} 个样本")
        
        # 测试第一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n第一个样本:")
            print(f"  session_id: {sample['session_id']}")
            print(f"  acoustic shape: {sample['acoustic'].shape if sample['acoustic'] is not None else None}")
            print(f"  embeds shape: {sample['embeds'].shape if sample['embeds'] is not None else None}")
            print(f"  exertion_level: {sample['exertion_level']}")
        
        # 测试数据加载器
        from torch.utils.data import DataLoader
        from src.data.loader import collate_multi_feature_batch
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_multi_feature_batch
        )
        
        print(f"\n=== 测试数据加载器 ===")
        for batch_idx, batch in enumerate(dataloader):
            print(f"批次 {batch_idx}:")
            print(f"  acoustic shape: {batch['acoustic'].shape if batch['acoustic'] is not None else None}")
            print(f"  embeds shape: {batch['embeds'].shape if batch['embeds'] is not None else None}")
            print(f"  exertion_levels shape: {batch['exertion_levels'].shape if batch['exertion_levels'] is not None else None}")
            break
        
        # 测试模型
        print(f"\n=== 测试模型 ===")
        model_config = config.get_model_config()
        print(f"模型配置: {model_config}")
        
        model = create_model(model_config)
        print(f"模型输入维度: {model.input_dim}")
        
        # 测试前向传播
        if batch['acoustic'] is not None and batch['embeds'] is not None:
            try:
                with torch.no_grad():
                    outputs = model(mfcc=batch['acoustic'], wav2vec2=batch['embeds'])
                    print(f"模型输出 shape: {outputs.shape}")
                    print("✅ 模型前向传播成功！")
            except Exception as e:
                print(f"❌ 模型前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ 特征加载测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_loading()
