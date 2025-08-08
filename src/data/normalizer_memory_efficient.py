#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存友好的特征标准化器
避免一次性加载所有特征到内存中
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import gc


class MemoryEfficientFeatureNormalizer:
    """
    内存友好的特征标准化器
    通过分批处理和增量计算来减少内存使用
    """
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.scalers = {}
        self.feature_names = []
        self.is_fitted = False
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def fit_transform_incremental(self, dataset, train_indices: List[int], 
                                 val_indices: List[int], 
                                 test_indices: Optional[List[int]] = None,
                                 batch_size: int = 100) -> Tuple[Dict, ...]:
        """
        增量式特征标准化，减少内存使用
        
        Args:
            dataset: 数据集对象
            train_indices: 训练集索引
            val_indices: 验证集索引
            test_indices: 测试集索引（可选）
            batch_size: 批处理大小
            
        Returns:
            标准化后的特征字典元组 (train_features, val_features, test_features)
        """
        print("开始内存友好的特征标准化...")
        
        # 获取特征名称
        sample = dataset[0]
        feature_names = []
        if sample['acoustic'] is not None:
            feature_names.append('mfcc')
        if sample['embeds'] is not None:
            feature_names.append('wav2vec2')
        
        print(f"特征名称: {feature_names}")
        
        # 初始化标准化器
        for feature_name in feature_names:
            self.scalers[feature_name] = StandardScaler()
        
        self.feature_names = feature_names
        
        # 分批拟合训练集
        print("分批拟合训练集特征...")
        self._fit_incremental(dataset, train_indices, batch_size)
        
        # 分批转换所有数据集
        print("分批转换训练集...")
        train_features = self._transform_incremental(dataset, train_indices, batch_size)
        
        print("分批转换验证集...")
        val_features = self._transform_incremental(dataset, val_indices, batch_size)
        
        test_features = None
        if test_indices is not None:
            print("分批转换测试集...")
            test_features = self._transform_incremental(dataset, test_indices, batch_size)
        
        # 保存标准化器参数
        self.save_scalers()
        
        self.is_fitted = True
        print("内存友好的特征标准化完成")
        
        return train_features, val_features, test_features
    
    def _fit_incremental(self, dataset, indices: List[int], batch_size: int):
        """增量式拟合标准化器"""
        for feature_name in self.feature_names:
            print(f"  拟合特征: {feature_name}")
            
            # 分批处理
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                # 提取批次特征
                batch_features = []
                for idx in batch_indices:
                    sample = dataset[idx]
                    if feature_name == 'mfcc' and sample['acoustic'] is not None:
                        batch_features.append(sample['acoustic'])
                    elif feature_name == 'wav2vec2' and sample['embeds'] is not None:
                        batch_features.append(sample['embeds'])
                
                if batch_features:
                    # 重塑特征用于拟合
                    batch_array = np.array(batch_features)
                    batch_reshaped = batch_array.reshape(-1, batch_array.shape[-1])
                    
                    # 增量拟合
                    if i == 0:
                        self.scalers[feature_name].partial_fit(batch_reshaped)
                    else:
                        self.scalers[feature_name].partial_fit(batch_reshaped)
                
                # 清理内存
                del batch_features
                gc.collect()
    
    def _transform_incremental(self, dataset, indices: List[int], batch_size: int) -> Dict[str, np.ndarray]:
        """增量式转换特征"""
        features = {}
        
        for feature_name in self.feature_names:
            print(f"  转换特征: {feature_name}")
            
            all_transformed = []
            
            # 分批处理
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                # 提取批次特征
                batch_features = []
                for idx in batch_indices:
                    sample = dataset[idx]
                    if feature_name == 'mfcc' and sample['acoustic'] is not None:
                        batch_features.append(sample['acoustic'])
                    elif feature_name == 'wav2vec2' and sample['embeds'] is not None:
                        batch_features.append(sample['embeds'])
                
                if batch_features:
                    # 重塑特征用于转换
                    batch_array = np.array(batch_features)
                    original_shape = batch_array.shape
                    batch_reshaped = batch_array.reshape(-1, batch_array.shape[-1])
                    
                    # 转换
                    transformed_reshaped = self.scalers[feature_name].transform(batch_reshaped)
                    transformed = transformed_reshaped.reshape(original_shape)
                    
                    all_transformed.append(transformed)
                
                # 清理内存
                del batch_features
                gc.collect()
            
            if all_transformed:
                features[feature_name] = np.concatenate(all_transformed, axis=0)
                print(f"    {feature_name} 形状: {features[feature_name].shape}")
        
        return features
    
    def save_scalers(self):
        """保存标准化器参数"""
        scaler_params = {}
        
        for feature_name, scaler in self.scalers.items():
            scaler_params[feature_name] = {
                'mean_': scaler.mean_.tolist(),
                'scale_': scaler.scale_.tolist(),
                'var_': scaler.var_.tolist()
            }
        
        params_path = os.path.join(self.save_dir, 'scaler_params.json')
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(scaler_params, f, indent=2, ensure_ascii=False)
        
        print(f"标准化器参数已保存到 {params_path}")
    
    def load_scalers(self):
        """加载标准化器参数"""
        params_path = os.path.join(self.save_dir, 'scaler_params.json')
        
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"标准化器参数文件不存在: {params_path}")
        
        with open(params_path, 'r', encoding='utf-8') as f:
            scaler_params = json.load(f)
        
        for feature_name, params in scaler_params.items():
            scaler = StandardScaler()
            scaler.mean_ = np.array(params['mean_'])
            scaler.scale_ = np.array(params['scale_'])
            scaler.var_ = np.array(params['var_'])
            self.scalers[feature_name] = scaler
        
        self.feature_names = list(self.scalers.keys())
        self.is_fitted = True
        
        print(f"标准化器参数已从 {params_path} 加载")


def normalize_dataset_features_memory_efficient(dataset, train_indices: List[int], 
                                              val_indices: List[int], 
                                              test_indices: Optional[List[int]] = None, 
                                              save_dir: str = "data/normalization",
                                              batch_size: int = 100) -> Tuple[Dict, ...]:
    """
    内存友好的数据集特征标准化
    
    Args:
        dataset: 数据集对象
        train_indices: 训练集索引
        val_indices: 验证集索引
        test_indices: 测试集索引（可选）
        save_dir: 标准化参数保存目录
        batch_size: 批处理大小
        
    Returns:
        标准化后的特征字典元组 (train_features, val_features, test_features)
    """
    print("开始内存友好的数据集特征标准化...")
    
    # 创建内存友好的标准化器
    normalizer = MemoryEfficientFeatureNormalizer(save_dir)
    
    # 增量式标准化
    normalized_train, normalized_val, normalized_test = normalizer.fit_transform_incremental(
        dataset, train_indices, val_indices, test_indices, batch_size
    )
    
    return normalized_train, normalized_val, normalized_test
